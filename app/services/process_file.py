import pandas as pd
from tabulate import tabulate
import os
import base64
import time
from mistralai import Mistral
from dotenv import load_dotenv
import re
from datetime import datetime
# Load environment variables
load_dotenv()
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from markitdown import MarkItDown
from mrkdwn_analysis import MarkdownAnalyzer
from app.services.file_profiler_agent import FileAnalyzer

MISTRAL_OCR_KEY = os.getenv("MISTRAL_OCR_KEY")
client = Mistral(api_key=MISTRAL_OCR_KEY)
output_dir = "markdown_output" 
os.makedirs(output_dir, exist_ok=True) 

def extract_markdown_from_file(file_path):

    if not file_path:
        print(f'Error: Failed to load document {file_path}')
        return None, None

    with open(file_path, "rb") as f:
        encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
    document = {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{encoded_pdf}"
    }
    try:
        ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document, include_image_base64=False)
        time.sleep(1)  # prevent rate limiting
        pages = ocr_response.pages if hasattr(ocr_response, "pages") else ocr_response
        all_markdown = ""
        for i, page in enumerate(pages):
            markdown = page.markdown
            all_markdown += markdown + "\n\n"
        
        file_name = os.path.basename(file_path)
        base_name, _ = os.path.splitext(file_name)
        output_file_path = os.path.join(output_dir, f"{base_name}.md")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(all_markdown)
        
        print(f"Extracted all markdown from {file_path} to {output_file_path}")
        return output_file_path, all_markdown  # Return a single entry

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def parse_csv_with_llm(csv_text: str) -> dict:
    """
    Uses an LLM to parse raw CSV text into a structured dictionary.

    Args:
        csv_text: A string containing the raw CSV data.

    Returns:
        A dictionary with keys 'data_start_row', 'headers', and 'data'.
    """
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # 2. Define the output parser to enforce JSON format
    parser = JsonOutputParser()

    # 3. Create the GENERALIZED prompt template
    prompt = PromptTemplate(
        template="""
        You are an expert data parsing assistant. Your task is to analyze raw text representing a CSV file and extract its primary data table.

        Follow these general rules:
        1.  **Ignore Metadata**: Automatically detect and ignore any file titles, descriptions, or metadata at the top, and any footnotes or summary notes at the bottom.
        2.  **Identify Headers**: Locate the header row(s). Headers might be on a single line or spread across multiple lines.
        3.  **Consolidate Multi-line Headers**: If headers are spread across multiple rows, intelligently merge them. A common pattern is a broad category in one row (e.g., 'Demographics') with specific metrics in the row below (e.g., 'Age', 'City'). Your output should be a single list of combined headers (e.g., 'Demographics_Age', 'Demographics_City').
        4.  **Clean Headers**: Convert the final headers into clean, database-friendly names. Remove special characters and replace spaces with underscores.
        5.  **Extract Data Rows**: Extract all the contiguous data rows that belong to the identified table.
        6.  **Provide Structured Output**: Return your findings as a single JSON object.

        {format_instructions}

        Here is the raw text to parse:
        ```
        {csv_input}
        ```
        """,
        input_variables=["csv_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 4. Create and invoke the LangChain chain
    chain = prompt | llm | parser
    print("Calling LLM with generalized prompt...")
    parsed_result = chain.invoke({"csv_input": csv_text})
    print("LLM parsing complete.")
    return parsed_result

def create_markdown_table(data):
    """Formats the parsed data into a Markdown table."""
    if not data or 'headers' not in data or 'data' not in data:
        return "Could not parse data into a table."
        
    headers = data['headers']
    rows = data['data']

    header_md = "| " + " | ".join(map(str, headers)) + " |"
    separator_md = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    rows_md = []
    for row in rows:
        # Ensure row has the same number of columns as headers
        # and convert all cells to strings for the join operation
        str_row = [str(cell) for cell in row]
        if len(str_row) != len(headers):
            # Pad row if it's shorter than the header
            str_row.extend([''] * (len(headers) - len(str_row)))
        rows_md.append("| " + " | ".join(str_row) + " |")

    return "\n".join([header_md, separator_md] + rows_md)

def csv_to_markdown_file(markdown_path: str):
    # Ensure input file exists
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(f"CSV file not found: {markdown_path}")
    
    # Step 1: Read CSV content
    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    parsed_data = parse_csv_with_llm(markdown_text)
    
    # Step 3: Create Markdown table
    markdown_content = create_markdown_table(parsed_data)
    
    # Step 4: Save Markdown output
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"✅ Markdown table saved to {markdown_path}")
    return markdown_content


def categorize_structure(md):
    total_lines = 0
    table_lines = 0

    lines = md.split('\n')
    total_lines += len(lines)
    table_lines += sum(1 for line in lines if '|' in line)

    if total_lines == 0:
        return "Unstructured"

    ratio = table_lines / total_lines
    if ratio > 0.5:
        return "Structured"
    elif 0.2 <= ratio <= 0.5:
        return "Semi-Structured"
    return "Unstructured"

def check_parse_quality(markdown):
    lines = markdown.split('\n')
    table_lines = [line for line in lines if '|' in line]
    valid_rows = [line for line in table_lines if line.count('|') >= 3]

    if len(valid_rows) >= 5:
        return 3  # High quality
    elif len(valid_rows) >= 3:
        return 2
    elif len(valid_rows) >= 1:
        return 1
    return 0

def check_format_consistency(markdown):
    table_lines = [line for line in markdown.split('\n') if '|' in line]
    header_lines = [line for line in markdown.split('\n') if line.strip().startswith('#')]
    list_lines = [line for line in markdown.split('\n') if line.strip().startswith(('-', '*'))]

    count = sum([
        1 if len(table_lines) > 1 else 0,
        1 if len(header_lines) > 0 else 0,
        1 if len(list_lines) > 1 else 0,
    ])
    return count  # 0 to 3 scale

def analyze_table(table_text):
    """
    Analyzes a single Markdown table to determine its complexity.

    Complexity Levels:
    - 1 (Easy): A standard table with a consistent number of columns.
    - 2 (Medium): A table with merged cells (indicated by fewer columns in a row
                  than in the header).
    - 3 (High): A table containing another table within one of its cells.

    Args:
        table_text (str): The string content of a single Markdown table.

    Returns:
        int: The complexity level (1, 2, or 3).
    """
    lines = [line.strip() for line in table_text.strip().split('\n')]
    if len(lines) < 2:
        return 1 # Not a valid table, treat as simple.

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    # --- Validate separator line ---
    # A valid separator must contain '---' and pipes.
    if not re.match(r'^\s*\|?.*-.*\|?\s*$', separator):
        # If the separator is invalid, it's not a standard table.
        # We can treat this as a simple case or handle as an error.
        # For this purpose, we'll consider it not a table.
        return 1

    # --- Determine header column count ---
    # Counts the number of cells by splitting the stripped header.
    # We strip leading/trailing pipes and then split by the middle pipes.
    header_cols = [col.strip() for col in header.strip().strip('|').split('|')]
    num_header_cols = len(header_cols)

    has_merged_cells = False
    is_highly_complex = False

    # --- Analyze each data row ---
    for row in rows:
        # Check for nested tables (high complexity)
        # A simple heuristic: does a cell itself contain table-like syntax?
        # This checks for a pipe-separated value within a cell.
        if re.search(r'\|.*\|.*\|', row):
             is_highly_complex = True
             break # Found highest complexity, no need to check further

        # --- Check for merged cells (medium complexity) ---
        # We count columns in the same way as the header.
        row_cols = [col.strip() for col in row.strip().strip('|').split('|')]
        num_row_cols = len(row_cols)

        # If a row has fewer columns than the header, we assume a merged cell.
        if num_row_cols > 0 and num_row_cols < num_header_cols:
            has_merged_cells = True

    # --- Return final complexity score for the table ---
    if is_highly_complex:
        return 3
    if has_merged_cells:
        return 2
    return 1

def check_file_complexity(markdown_text):
    table_pattern = re.compile(
        r"((?:\|.*\|\s*\n)+)"       # 1+ header lines
        r"(\s*\|(?::?-+:?\|)+)\s*\n" # Separator line
        r"((?:\|.*\|\s*\n?)+)",      # 1+ data row lines
        re.MULTILINE
    )

    tables = table_pattern.findall(markdown_text)

    if not tables:
        return 0

    max_complexity = 0

    # Iterate through all found tables and find the max complexity
    for i, table_parts in enumerate(tables):
        # Reconstruct the full table text from the matched parts
        full_table_text = "".join(table_parts)
        complexity = analyze_table(full_table_text)
        if complexity > max_complexity:
            max_complexity = complexity

    return max_complexity    

def table_complexity(analyzer: MarkdownAnalyzer) -> int:
    """
    Analyzes the complexity of tables from MarkdownAnalyzer output.

    Rules:
    1 = Easy: Header regular + Data regular
    2 = Medium: Header irregular only (data regular)
    3 = Hard: Header irregular AND Data irregular

    Header irregularity: Header contains 'Unnamed' (case-insensitive).
    Data irregularity: Any data row with column count mismatch.
    """

    tables_data = analyzer.identify_tables()
    if not tables_data.get("Table"):
        return 1  # No tables = easy

    overall_score = 1

    for tbl in tables_data["Table"]:
        header = tbl.get("header", [])
        rows = tbl.get("rows", [])
        print(header)
        if not header and not rows:
            continue

        # Expected columns = from first valid data row
        base_cols = None
        for r in rows:
            if all(str(cell).strip().lower() not in ["nan", "", "none"] for cell in r):
                base_cols = len(r)
                break
        if base_cols is None:
            continue

        # Header irregular if contains 'Unnamed'
        header_irregular = any("unnamed" in str(h).lower() or len(h) == 0 for h in header)

        # Data irregular if any row's col count != expected
        data_irregular = any(len(r) != base_cols for r in rows)

        # Determine score for this table
        if not header_irregular and not data_irregular:
            score = 1
        elif header_irregular and not data_irregular:
            score = 2
        elif header_irregular and data_irregular:
            score = 3
        else:
            score = 2  # Data irregular but header regular
        print(score)
        overall_score = max(overall_score, score)
    
    return overall_score

def csv_to_markdown_table(file_path):
    df = pd.read_csv(file_path)
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    all_markdown = markdown_table
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    output_file_path = os.path.join(output_dir, f"{base_name}.md")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(all_markdown)
    return output_file_path, markdown_table

def extract_markdown_from_excel_csv(file_path):
    md = MarkItDown(docintel_endpoint="<document_intelligence_endpoint>")
    result = md.convert(file_path)
    all_markdown = result.text_content
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    output_file_path = os.path.join(output_dir, f"{base_name}.md")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(all_markdown)
    return output_file_path, all_markdown


def process_pdf(file_path):
    file_name = os.path.basename(file_path)
    base_name, file_type = os.path.splitext(file_name)
    markdown_path = os.path.join(output_dir, f"{base_name}.md")

    markdown_result = None

    markdown_file_path = None

    if os.path.exists(markdown_path):
        print(f"Using cached markdown file: {markdown_path}")
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_result = f.read()
        markdown_file_path = markdown_path
    else:
        if 'xl' in file_type or 'csv' in file_type:
            markdown_file_path, markdown_result = extract_markdown_from_excel_csv(file_path)
        else:
            markdown_file_path, markdown_result = extract_markdown_from_file(file_path)
        if not markdown_result:
            return None

    quality_scores = check_parse_quality(markdown_result)
    analyzer = MarkdownAnalyzer(markdown_file_path)
    complexity_score = table_complexity(analyzer)
    analyzer_instance = FileAnalyzer()
    analysis = analyzer_instance.analyze_single_file(file_path)
    structure = analysis.get('content_type')

    print(analysis, structure)
    if structure == 'structured':
        structure = "Structured"
    elif structure == 'semi-structured':
        structure = "Semi-Structured"
    else:
        structure = "Unstructured"

    return {
        "file": file_path,
        "structure": structure,
        "avg_parse_quality": round(quality_scores, 2),
        "complexity_score": (3 - round(complexity_score, 2)) + 1,
        "analysis": analysis
    }