#!/usr/bin/env python3
"""
file_profiler_agent.py

A focused file profiler for analyzing single files. It uses PostgreSQL for tracking,
extracts text from various document types, and leverages an LLM for analysis.

This version is streamlined to focus on the single-file analysis function.

Usage (as a library):
    from file_profiler_pg import FileAnalyzer
    analyzer = FileAnalyzer()
    result = analyzer.analyze_single_file('/path/to/your/file.pdf')
    print(result)

Usage (from command line):
    python file_profiler_pg.py analyze-file --file-path /path/to/your/file.pdf

Requirements:
    pip install pymupdf pandas python-docx openai psycopg2-binary
"""
import os
import sys
import time
import json
import argparse
import hashlib
import traceback
from pathlib import Path

# Third-party libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import docx
except ImportError:
    docx = None

# Postgres
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

# OpenAI
try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
except ImportError:
    OpenAI = None
from dotenv import load_dotenv
load_dotenv()
# ----------------- CONFIG -----------------
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = int(os.getenv("PG_PORT", 5432))
DB_NAME = os.getenv("PG_DB", "file_analysis")
DB_USER = os.getenv("PG_USER", "postgres")
DB_PASS = os.getenv("PG_PASS", "admin")
DB_MIN_POOL = int(os.getenv("PG_POOL_MIN", 1))
DB_MAX_POOL = int(os.getenv("PG_POOL_MAX", 8))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # It's better to not hardcode keys
print(OPENAI_API_KEY)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

DEFAULT_MAX_RPM = 3
DEFAULT_TEXT_LIMIT = 6000
MAX_RETRIES = 5
INITIAL_BACKOFF = 8.0

SUPPORTED_EXTS = {".pdf", ".xlsx", ".xls", ".xlsm", ".docx"}

# ----------------- FILE ANALYZER CLASS -----------------

class FileAnalyzer:
    """Encapsulates file profiling logic."""

    def __init__(self):
        """Initializes the analyzer and the database connection pool."""
        self._conn_pool = None
        self.init_db()
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI and OPENAI_API_KEY else None
        self._last_call_ts = 0.0

    def get_db_pool(self):
        """Creates and returns a threaded database connection pool."""
        if self._conn_pool is None:
            self._conn_pool = ThreadedConnectionPool(
                minconn=DB_MIN_POOL, maxconn=DB_MAX_POOL,
                host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
            )
        return self._conn_pool

    def init_db(self):
        """Initializes the database schema if it doesn't exist."""
        conn = None
        try:
            conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id BIGSERIAL PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT,
                    file_size BIGINT,
                    extract_status TEXT DEFAULT 'pending', -- pending / done / error
                    extract_error TEXT,
                    extracted_text TEXT,
                    analyzed BOOLEAN DEFAULT FALSE,
                    analysis_json JSONB,
                    analysis_error TEXT,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash);")
            conn.commit()
            cur.close()
        finally:
            if conn:
                conn.close()

    def _compute_file_hash(self, path: Path, chunk_size: int = 65536) -> str:
        """Computes the SHA256 hash of a file."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _extract_text(self, path: Path) -> (str, str):
        """Extracts text from a file, returns (text, error_string)."""
        try:
            ext = path.suffix.lower()
            if ext == ".pdf":
                txt = self._extract_pdf_text(path)
            elif ext in {".xlsx", ".xls", ".xlsm"}:
                txt = self._extract_excel_text(path)
            elif ext == ".docx":
                txt = self._extract_docx_text(path)
            else:
                txt = "[Unsupported file type]"

            if txt.startswith("[") and "Error" in txt:
                return "", txt
            return txt, ""
        except Exception:
            return "", f"Unexpected extraction error: {traceback.format_exc()}"

    def _extract_pdf_text(self, path: Path) -> str:
        if fitz is None:
            return "[PDF extraction error: pymupdf not installed]"
        try:
            with fitz.open(str(path)) as doc:
                return "\n".join(page.get_text("text").strip() for page in doc).strip()
        except Exception as e:
            return f"[PDF Extraction Error: {e}]"

    def _extract_excel_text(self, path: Path) -> str:
        if pd is None:
            return "[Excel extraction error: pandas/openpyxl not installed]"
        try:
            sheets = pd.read_excel(path, sheet_name=None, dtype=str)
            parts = [f"--- Sheet: {name} ---\n{df.to_csv(index=False)}" for name, df in sheets.items()]
            return "\n".join(parts).strip()
        except Exception as e:
            return f"[Excel Extraction Error: {e}]"

    def _extract_docx_text(self, path: Path) -> str:
        if docx is None:
            return "[DOCX extraction error: python-docx not installed]"
        try:
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            return f"[DOCX Extraction Error: {e}]"

    def _throttle(self, max_rpm: int):
        """Ensures the rate of API calls does not exceed max_rpm."""
        min_gap = 60.0 / max(1, max_rpm)
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._last_call_ts = time.time()

    def _classify_with_llm(self, text: str, file_name: str, text_limit: int):
        """Sends text to an LLM for classification and returns a JSON object."""
        if not self.client:
            return {"ok": False, "error": "OpenAI client not initialized. Check API key."}

        prompt = f"""
        You are an expert file diagnoser. Return ONLY a compact JSON object with keys:
        - file_name
        - content_type: one of ["structured","semi-structured","unstructured"].
            * structured → Tables, databases, spreadsheets, CSVs, machine-readable formats with consistent schema.
            * semi-structured → Documents mixing tables + text (e.g., reports with both prose and data tables).
            * unstructured → Pure text, paragraphs, scanned documents, or free-form notes without clear schema.
        - domain: broad domain (e.g., Finance, Healthcare, Legal)
        - subdomain: short tag
        - intents: Specific intents (e.g., India GST Collections, State of Assam econometrics)
        - publishing_authority: Publishing organization or committee
        - published_date: Fiscal year or month of reference
        - period_of_reference: Date range within the document
        - brief_summary: A concise summary (<= 3 sentences)
        - document_size: File size in KB or MB

        Rules:
        - Output valid JSON and nothing else.

        File: {file_name}
        Content:
        {text[:text_limit]}
        """.strip()

        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content
                return {"ok": True, "json": json.loads(content)}
            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
                print(f"API Error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {backoff}s...")
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    return {"ok": False, "error": f"OpenAI API error after retries: {e}", "traceback": traceback.format_exc()}
            except json.JSONDecodeError as e:
                 return {"ok": False, "error": f"JSON parse error: {e}", "raw_output": content}
            except Exception as e:
                return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

    def analyze_single_file(self, file_path: str, text_limit: int = DEFAULT_TEXT_LIMIT, max_rpm: int = DEFAULT_MAX_RPM) -> dict:
        """
        Analyzes a single file by extracting its text and sending it to an LLM.

        Args:
            file_path: The absolute path to the file.
            text_limit: The maximum number of characters to send for analysis.
            max_rpm: The maximum requests per minute for the LLM API.

        Returns:
            A dictionary containing the analysis result or an error message.
        """
        p = Path(file_path)
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            return {"error": "File does not exist or is not a supported type.", "file_path": file_path}

        # 1. Extract text
        extracted_text, err = self._extract_text(p)
        if err:
            return {"error": f"Extraction failed: {err}", "file_path": file_path}

        # 2. Throttle and Analyze
        self._throttle(max_rpm)
        analysis_result = self._classify_with_llm(extracted_text, p.name, text_limit)

        # 3. Store results in DB for tracking
        conn = self.get_db_pool().getconn()
        try:
            with conn.cursor() as cur:
                file_hash = self._compute_file_hash(p)
                file_size = p.stat().st_size
                if analysis_result.get("ok"):
                    cur.execute("""
                        INSERT INTO files (file_path, file_hash, file_size, extract_status, extracted_text, analyzed, analysis_json, updated_at)
                        VALUES (%s, %s, %s, 'done', %s, TRUE, %s, NOW())
                        ON CONFLICT (file_path) DO UPDATE SET
                            file_hash = EXCLUDED.file_hash, file_size = EXCLUDED.file_size,
                            extract_status = 'done', extracted_text = EXCLUDED.extracted_text,
                            analyzed = TRUE, analysis_json = EXCLUDED.analysis_json,
                            analysis_error = NULL, updated_at = NOW();
                    """, (file_path, file_hash, file_size, extracted_text, json.dumps(analysis_result['json'])))
                else:
                    cur.execute("""
                        INSERT INTO files (file_path, file_hash, file_size, extract_status, extracted_text, analyzed, analysis_error, updated_at)
                        VALUES (%s, %s, %s, 'done', %s, TRUE, %s, NOW())
                        ON CONFLICT (file_path) DO UPDATE SET
                            file_hash = EXCLUDED.file_hash, file_size = EXCLUDED.file_size,
                            extract_status = 'done', extracted_text = EXCLUDED.extracted_text,
                            analyzed = TRUE, analysis_json = NULL,
                            analysis_error = EXCLUDED.analysis_error, updated_at = NOW();
                    """, (file_path, file_hash, file_size, extracted_text, json.dumps(analysis_result)))
                conn.commit()
        finally:
            self.get_db_pool().putconn(conn)

        return analysis_result

# ----------------- CLI -----------------
def make_parser():
    p = argparse.ArgumentParser(description="Analyzes a single file and returns a JSON analysis.")
    p.add_argument("--file-path", required=True, help="Path to the file to analyze")
    p.add_argument("--max-rpm", type=int, default=DEFAULT_MAX_RPM, help="Max requests per minute for LLM API")
    p.add_argument("--text-limit", type=int, default=DEFAULT_TEXT_LIMIT, help="Max characters to send to LLM")
    return p

def main():
    parser = make_parser()
    args = parser.parse_args()
    analyzer = FileAnalyzer()
    print(f"Analyzing single file: {args.file_path}")
    result = analyzer.analyze_single_file(args.file_path, text_limit=args.text_limit, max_rpm=args.max_rpm)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # Example of how to use the new function directly from code.
    # Make sure to set your OPENAI_API_KEY environment variable.
    if len(sys.argv) == 1:
        print("This script is intended to be used with command-line arguments.")
        print("Example: python your_script_name.py --file-path /path/to/document.pdf")
        # You can uncomment the following lines to run an example directly
        if os.getenv("OPENAI_API_KEY"):
            analyzer_instance = FileAnalyzer()
            # Create a dummy file or point to an existing one
            dummy_file_path = "data/pdf-input/downloads/LFPR_April_2025_page_0_ZMYVL8FK9D.xlsx"
            if not Path(dummy_file_path).exists():
                print(f"Creating a dummy file: {dummy_file_path}")
                # Requires python-docx to be installed
                try:
                    from docx import Document
                    doc = Document()
                    doc.add_paragraph("This is a test document for the file analyzer.")
                    doc.save(dummy_file_path)
                    analysis = analyzer_instance.analyze_single_file(dummy_file_path)
                    print(json.dumps(analysis, indent=2))
                except Exception as e:
                    print(f"Could not create dummy docx file. Please install python-docx. Error: {e}")
            else:
                analysis = analyzer_instance.analyze_single_file(dummy_file_path)
                print(json.dumps(analysis, indent=2))
        else:
            print("OPENAI_API_KEY environment variable not set. Cannot run example.")
    else:
        main()
