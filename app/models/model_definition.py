from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional, Dict, Any

class QualityMetrics(BaseModel):
    """Defines the structure for data quality metrics."""
    parseAccuracy: float
    complexity: float

class AnalysisResult(BaseModel):
    """
    Defines the structure for the analysis object returned by the LLM.
    This model is flexible to handle potential variations in LLM output.
    """
    file_name: Optional[str] = None
    content_type: Optional[str] = None
    domain: Optional[str] = None
    subdomain: Optional[str] = None
    intents: Optional[Union[List[str], str]] = None
    publishing_authority: Optional[str] = None
    published_date: Optional[str] = None
    period_of_reference: Optional[str] = None
    brief_summary: Optional[str] = None
    document_size: Optional[str] = None
    # This field will capture any other keys the LLM might return
    extra_fields: Dict[str, Any] = {}
    # This field will capture error details if the analysis fails
    error: Optional[str] = None

class FileProcessingResult(BaseModel):
    """
    Defines the final response structure for each processed file.
    """
    fileName: str
    qualityMetrics: QualityMetrics
    classification: str
    analysis: AnalysisResult

class ColumnSchema(BaseModel):
    """Defines the schema for a single column in a structured table."""
    name: str
    type: str
    primary: Optional[bool] = False

class TableDetails(BaseModel):
    """Holds all the details for a single table extracted from a structured file."""
    tableName: str
    schema_details: List[ColumnSchema]
    rowsInserted: int
    sqlCommands: List[str]
    fileSelectorPrompt: Optional[str] = None

class StructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested structured file, supporting multiple tables."""
    type: Literal["structured"]
    tables: List[TableDetails]
 
class UnstructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested unstructured file."""
    type: Literal["unstructured"]
    collection: str
    chunksCreated: int
    embeddingsGenerated: int
    chunkingMethod: str
    embeddingModel: str

# A union of all possible ingestion detail types
IngestionDetails = Union[StructuredIngestionDetails, UnstructuredIngestionDetails]

class FileIngestionResult(BaseModel):
    """Represents the result of processing a single file."""
    fileName: str
    fileSize: int
    status: Literal["success", "failed"]
    ingestionDetails: Optional[Union[IngestionDetails, List[IngestionDetails]]] = None
    error: Optional[str] = None

class IngestionResponse(BaseModel):
    """The final response object returned by the API."""
    results: List[FileIngestionResult]
