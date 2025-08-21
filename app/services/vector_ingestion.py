import os
import base64
import time
import numpy as np
import pandas as pd
from mistralai import Mistral
from dateutil import parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, db, utility, FieldSchema, DataType, Collection, CollectionSchema, MilvusException
from dotenv import load_dotenv
import re
from datetime import datetime
from pymilvus.exceptions import ParamError
from langdetect import detect
from PyPDF2 import PdfReader, PdfWriter

from app.models.model_definition import QualityMetrics, AnalysisResult, FileProcessingResult, ColumnSchema, TableDetails, StructuredIngestionDetails, SemiStructuredIngestionDetails, UnstructuredIngestionDetails, IngestionDetails, FileIngestionResult, IngestionResponse

# ==============================================================================
# 2. Environment Setup and Milvus Connection
# ==============================================================================

# Load environment variables
load_dotenv()

MISTRAL_OCR_KEY = os.getenv("MISTRAL_OCR_KEY")
client = Mistral(api_key=MISTRAL_OCR_KEY)

# Connect to Milvus
conn = connections.connect(host='localhost', port=19530)
db_name = 'test_db'
if db_name not in db.list_database():
    db.create_database(db_name)
db.using_database(db_name)

# Define schema and collection for Milvus
collection_name = 'vector_ingestion'
if not utility.has_collection(collection_name):
    id_field = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True)
    source_field = FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=255)
    page_field = FieldSchema(name='page', dtype=DataType.INT64)
    category_field = FieldSchema(name='category', dtype=DataType.VARCHAR, max_length=50)
    embedding_field = FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=768)
    content_field = FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=8192)
    reference_field = FieldSchema(name='reference', dtype=DataType.VARCHAR, max_length=255)
    date_field = FieldSchema(name='date', dtype=DataType.VARCHAR, max_length=50)
    url_field = FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=1000)

    schema = CollectionSchema(fields=[
        id_field, source_field, page_field, category_field,
        embedding_field, content_field, reference_field, date_field, url_field
    ])
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': 'COSINE',
        'index_type': 'HNSW',
        'params': {'M': 16, 'efConstruction': 200}
    }
    collection.create_index(field_name='embeddings', index_params=index_params)
else:
    collection = Collection(name=collection_name)

collection.load()

# Load Sentence Transformer model
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

def split_pdf_if_large(file_path, max_size_mb=50, max_pages_per_part=20):
    if os.path.getsize(file_path) / (1024 * 1024) <= max_size_mb:
        return [file_path]
    
    output_dir = os.path.join(os.path.dirname(file_path), 'split_parts')
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(file_path)
    total_pages, split_files = len(reader.pages), []
    
    for i in range(0, total_pages, max_pages_per_part):
        writer = PdfWriter()
        for j in range(i, min(i + max_pages_per_part, total_pages)):
            writer.add_page(reader.pages[j])
        
        part_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_part_{i//max_pages_per_part + 1}.pdf")
        with open(part_path, 'wb') as f:
            writer.write(f)
        split_files.append(part_path)
    return split_files

def filter_english_lines(text):
    english_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped: continue
        if '[SECTION]' in stripped:
            english_lines.append(stripped)
            continue
        try:
            if detect(stripped) == 'en':
                english_lines.append(stripped)
        except:
            continue
    return '\n'.join(english_lines)

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def extract_markdown_from_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
        document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded_pdf}"}
        ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document, include_image_base64=False)
        time.sleep(1)
        pages = ocr_response.pages if hasattr(ocr_response, "pages") else ocr_response
        return [(i + 1, page.markdown) for i, page in enumerate(pages)]
    except Exception as e:
        print(f"Error processing {file_path} with OCR: {e}")
        raise

def content_aware_chunk(text, chunk_size=2500, chunk_overlap=200):
    text = re.sub(r'## (.*?)\n', r'[SECTION] \1\n', text)
    text = re.sub(r'# (.*?)\n', r'[SECTION] \1\n', text)
    chunks = re.split(r'(?=\[SECTION\])', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def extract_date_from_reference(reference):
    try:
        return parser.parse(reference, fuzzy=True).strftime('%B %Y')
    except (ValueError, TypeError):
        return "Unknown Date"

# ==============================================================================
# 4. Core Processing Logic
# ==============================================================================

def process_and_embed_pdf(file_paths, original_file_name, category, reference, date_str, url):
    """
    Processes PDF file parts, creates chunks and embeddings, and inserts them into Milvus.
    Returns the number of chunks successfully created and inserted.
    """
    all_page_markdown_list, page_offset = [], 0
    for path in file_paths:
        page_markdown_list = extract_markdown_from_pdf(path)
        if not page_markdown_list:
            raise ValueError(f"Failed to extract markdown from document part {path}")
        
        adjusted_pages = [(page_number + page_offset, markdown) for page_number, markdown in page_markdown_list]
        all_page_markdown_list.extend(adjusted_pages)
        page_offset += len(page_markdown_list)

    chunks, page_numbers, urls = [], [], []
    for page_number, markdown in all_page_markdown_list:
        english_text = filter_english_lines(markdown)
        if not english_text.strip(): continue
        
        page_chunks = content_aware_chunk(english_text)
        chunks.extend(page_chunks)
        page_numbers.extend([page_number] * len(page_chunks))
        urls.extend([url] * len(page_chunks))
    
    if not chunks:
        raise ValueError("No processable content found in the document.")
        
    final_embeddings, final_contents = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    for chunk, page in zip(chunks, page_numbers):
        combined = f'Content from {reference}. Page number: {page}. {chunk}'
        if len(combined) <= 8192:
            embedding = embedding_model.encode(combined)
            final_embeddings.append(l2_normalize(np.array(embedding)))
            final_contents.append(combined)
        else:
            # If a section is still too large, split it further
            sub_chunks = splitter.split_text(chunk)
            for sub_chunk in sub_chunks:
                combined_sub = f'Content from {reference}. Page number: {page}. {sub_chunk}'
                if len(combined_sub) <= 8192:
                    embedding = embedding_model.encode(combined_sub)
                    final_embeddings.append(l2_normalize(np.array(embedding)))
                    final_contents.append(combined_sub)

    if not final_contents:
        raise ValueError("Failed to generate any valid chunks for embedding.")
        
    # Prepare data for Milvus insertion
    num_chunks = len(final_contents)
    data_to_insert = [
        [original_file_name] * num_chunks,
        page_numbers[:num_chunks],  # Ensure lists are of the same length
        [category] * num_chunks,
        final_embeddings,
        final_contents,
        [reference] * num_chunks,
        [date_str] * num_chunks,
        urls[:num_chunks] # Ensure lists are of the same length
    ]
    
    try:
        collection.insert(data_to_insert)
        print(f'Successfully loaded {num_chunks} chunks for PDF: {original_file_name}')
        return num_chunks
    except (MilvusException, ParamError) as e:
        print(f'[MILVUS ERROR] for {original_file_name}: {e}')
        raise # Re-raise exception to be caught by the main handler

# ==============================================================================
# 5. Main Ingestion Function
# ==============================================================================

def ingest_unstructured_file(
    file_path: str,
    category: str,
    reference: str,
    url: str
) -> FileIngestionResult:
    """
    Main function to handle the ingestion of a single unstructured file.
    
    Args:
        file_path: The local path to the file.
        category: The category to assign to the document chunks.
        reference: A reference string for the document (e.g., source name).
        url: The source URL of the document.
        
    Returns:
        A FileIngestionResult object with the status and details of the operation.
    """
    if not os.path.exists(file_path):
        return FileIngestionResult(
            fileName=os.path.basename(file_path),
            fileSize=0,
            status="failed",
            error="File not found at the specified path."
        )

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    try:
        # 1. Extract date from reference metadata
        date_str = extract_date_from_reference(reference)
        
        # 2. Split PDF if it's too large for the OCR API
        split_files = split_pdf_if_large(file_path)
        
        # 3. Process the file parts, create embeddings, and insert into Milvus
        chunks_created = process_and_embed_pdf(
            file_paths=split_files,
            original_file_name=file_name,
            category=category,
            reference=reference,
            date_str=date_str,
            url=url
        )
        
        # 4. Create the detailed success response
        ingestion_details = UnstructuredIngestionDetails(
            type="unstructured",
            collection=collection_name,
            chunksCreated=chunks_created,
            embeddingsGenerated=chunks_created,
            chunkingMethod="Content Aware Section Splitting",
            embeddingModel=embedding_model_name
        )
        
        return FileIngestionResult(
            fileName=file_name,
            fileSize=file_size,
            status="success",
            ingestionDetails=ingestion_details
        )
        
    except Exception as e:
        # 5. Create the failure response
        return FileIngestionResult(
            fileName=file_name,
            fileSize=file_size,
            status="failed",
            error=f"An unexpected error occurred: {str(e)}"
        )