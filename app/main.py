# main.py
import os
import shutil
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.services.process_file import process_pdf, csv_to_markdown_file
from app.services.file_handler import extract_markdown_tables, clean_markdown
from app.services.table_agent import ingest_markdown_table
from app.services.table_agent import ingest_markdown_table, ColumnSchema, TableDetails, StructuredIngestionDetails, SemiStructuredIngestionDetails, UnstructuredIngestionDetails, IngestionDetails, FileIngestionResult

from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional
import json

UPLOAD_DIRECTORY = "uploaded_files"
MARKDOWN_DIRECTORY = "markdown_output"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the application.
    This replaces the deprecated on_event("startup") and on_event("shutdown").
    """
    # --- Startup Event ---
    # Ensure the upload directory exists when the application starts.
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)
        print(f"Created directory: {UPLOAD_DIRECTORY}")
    yield
    # --- Shutdown Event ---
    # Add any cleanup tasks here if needed.
    print("Application shutdown complete.")

# --- App Initialization ---
# Create a FastAPI app instance
app = FastAPI(
    title="File Ingestion API",
    description="An API to receive and store user-uploaded files.",
)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://34.41.241.77:8071",
    "http://localhost:8071",
    "http://0.0.0.0:3000"
    "*"  # Using "*" is permissive; tighten this for production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QualityMetrics(BaseModel):
    """Defines the structure for data quality metrics."""
    parseAccuracy: int
    completeness: int
    complexity: int

class FileProcessingResult(BaseModel):
    """Defines the structure for the final processing result of a file."""
    fileName: str
    qualityMetrics: QualityMetrics
    classification: str

class IngestionResponse(BaseModel):
    """The final response object returned by the API."""
    results: List[FileIngestionResult]

# --- API Endpoints ---
@app.post("/ingest-files/", summary="Ingest and Store Files")
async def ingest_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")

    stored_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIRECTORY, str(file.filename))
        try:
            # Ensure the directory exists before writing the file
            if not os.path.exists(UPLOAD_DIRECTORY):
                os.makedirs(UPLOAD_DIRECTORY)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            stored_files.append(file.filename)
            print(f"Successfully stored file: {file.filename}")

        except Exception as e:
            print(f"Error storing file {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not save file: {file.filename}. Error: {e}"
            )
        finally:
            await file.close()

    return {
        "message": f"Successfully stored {len(stored_files)} file(s).",
        "stored_files": stored_files,
        "storage_path": os.path.abspath(UPLOAD_DIRECTORY)
    }

@app.post("/process-files", response_model=List[FileProcessingResult])
async def process_files(files: List[UploadFile] = File(...)):
    """
    Receives a list of files, processes each one to generate metrics
    and classification, and returns the results.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    results = []
    for file in files:
        if not file.filename:
            continue
        base_name = os.path.basename(file.filename)
        file_path = os.path.join(UPLOAD_DIRECTORY, str(base_name))

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file.filename}"
            )

        result = process_pdf(file_path)
        
        print("Raw result from process_pdf:", result)  # Inspect the result

        if not result:
            if not file.filename:
                continue
            base_name = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_DIRECTORY, str(base_name))

            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {file.filename}"
                )

            result = process_pdf(file_path)
            if not result:

                raise HTTPException(
                    status_code=500,
                    detail=f"Could not process file: {file.filename}"
                )
        
        try:
            quality_metrics = QualityMetrics(
                parseAccuracy=result.get("avg_parse_quality", 0),
                completeness=result.get("avg_completeness", 0),
                complexity=result.get("complexity_score", 0)
            )

            results.append(
                FileProcessingResult(
                    fileName=file.filename,
                    qualityMetrics=quality_metrics,
                    classification=result.get("structure", "Unknown")
                )
            )
        except Exception as e:
            print(f"Error creating FileProcessingResult for {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not create processing result for {file.filename}. Error: {e}"
            )

    return results

@app.post("/ingest/", response_model=IngestionResponse, tags=["Ingestion"])
async def start_ingestion_process(
    files: List[UploadFile] = File(...),
    file_details: str = Form(...),
    db_config: str = Form(...)  # Added db_config
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided for ingestion.")

    try:
        details_list = json.loads(file_details)
        print(f"Parsed file details: {details_list}")  # Debugging line
        db_config_data = json.loads(db_config) # Parsed db_config
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for file_details or db_config.")

    ingestion_results = []
    files_map = {file.filename: file for file in files}

    for details_data in details_list:
        if details_data.get("classification") == "Structured":
            filename = details_data.get("name")
            if not filename:
                continue

            filename_without_ext, file_type = os.path.splitext(str(filename))
            
            try:
                file_path = os.path.join(MARKDOWN_DIRECTORY, filename_without_ext+ ".md")
                print(f"Processing file: {file_path}")

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Processed markdown file not found: {file_path}")

                if details_data.get('qualityMetrics').get('complexity')>1:
                    if 'csv' in file_type or 'xl' in file_type:
                        with open(file_path, 'r', encoding='utf-8') as tf:
                            markdown_text = tf.read()
                        cleaned_md = clean_markdown(markdown_text)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(cleaned_md)

                table_files = extract_markdown_tables(file_path)
                print(f"Extracted {len(table_files)} tables from {file_path}")
                table_details = []
                for table_file in table_files:
                    with open(table_file, 'r', encoding='utf-8') as tf:
                        table_markdown = tf.read()
                    if details_data.get('qualityMetrics').get('complexity')>1:
                        table_markdown = csv_to_markdown_file(table_file)
                    table_ingest_result = ingest_markdown_table(table_markdown, str(filename), details_data.get('size', 0))
                    
                    print(f"Table ingestion result: {table_ingest_result}")
                    table_details.append(table_ingest_result)
                    

                structured_ingestion_details = StructuredIngestionDetails(type="structured", tables=table_details)
                ingestion_results.append(FileIngestionResult(
                        fileName=str(filename),
                        fileSize=details_data.get('size', 0),
                        status="success",
                        ingestionDetails=structured_ingestion_details,
                    ))

            except Exception as e:
                ingestion_results.append(
                    FileIngestionResult(
                        fileName=str(filename),
                        fileSize=details_data.get('size', 0),
                        status="failed",
                        error=f"An unexpected error occurred: {str(e)}"
                    )
                )
    result = IngestionResponse(results=ingestion_results)
    print(f"Ingestion results: {result}")
    return result
