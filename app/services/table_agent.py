import sqlite3
import pandas as pd
import io
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Optional

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

class StructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested structured file, supporting multiple tables."""
    type: Literal["structured"]
    tables: List[TableDetails]

class SemiStructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested semi-structured file."""
    type: Literal["semi-structured"]
    structuredData: dict
    unstructuredData: dict

class UnstructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested unstructured file."""
    type: Literal["unstructured"]
    collection: str
    chunksCreated: int
    embeddingsGenerated: int
    chunkingMethod: str
    embeddingModel: str

# A union of all possible ingestion detail types
IngestionDetails = Union[StructuredIngestionDetails, SemiStructuredIngestionDetails, UnstructuredIngestionDetails]

class FileIngestionResult(BaseModel):
    """Represents the result of processing a single file."""
    fileName: str
    fileSize: int
    status: Literal["success", "failed"]
    ingestionDetails: Optional[IngestionDetails] = None
    error: Optional[str] = None

# --- CONFIGURATION ---
# In a real application, you would initialize your actual LLM here.
# For example:
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

DB_PATH = "sql.db" # Use in-memory SQLite database for this example

# --- DATABASE HELPER FUNCTIONS ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def get_db_schema(conn):
    """
    Retrieves the schema of all tables in the database.
    This is crucial context for the LLM.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema_info = "Database Schema:\n"
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        schema_info += f"\nTable: {table_name}\n"
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            # col structure: (id, name, type, notnull, default_value, pk)
            schema_info += f"  - {col[1]} ({col[2]})\n"
    return schema_info.strip()

def get_table_schema(conn, table_name):
    """Retrieves the schema for a single table."""
    cursor = conn.cursor()
    schema_info = f"Table: {table_name}\n"
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        schema_info += f"  - {col[1]} ({col[2]})\n"
    return schema_info

def get_table_schema_json(conn, table_name):
    """Retrieves the schema for a single table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    column_schemas = {}
    for col in columns:
        column_schemas[col[1]] = col[2]
    return column_schemas

def execute_query(conn, query, params=None):
    """Executes a given SQL query."""
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    conn.commit()

def print_table_data(conn, table_name):
    """Prints all rows from a specified table for verification."""
    print(f"\n--- Current Data in '{table_name}' ---")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df.to_string())
    except Exception as e:
        print(f"Could not read from table '{table_name}': {e}")
    print("------------------------------------\n")


# --- MARKDOWN & DATAFRAME FUNCTIONS ---
import re
import mdpd

def parse_markdown_table(md_table_string):
    return mdpd.from_md(md_table_string)

# --- LLM-POWERED LOGIC ---

def get_matching_table_name(schema, df_columns, df_sample_rows):
    """
    Uses an LLM to determine if the new data can fit into an existing table.
    """
    print("\nStep 2: Asking LLM to find a matching table...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert database administrator. Your task is to determine if new data can be inserted into an existing table.

Here is the current database schema:
{schema}

Here are the headers and a few sample rows of the new data to be inserted:
Headers: {columns}
Sample Rows:
{sample_rows}

Does the structure and content of the new data strongly match one of the existing tables?
- A strong match means the columns are semantically equivalent and the data types are compatible.
- If a strong match is found, respond with ONLY the name of that table.
- If no strong match is found, respond with ONLY the word 'None'.
"""),
        ("human", "Based on the schema and new data, what is the matching table name?")
    ])

    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "schema": schema,
        "columns": ", ".join(df_columns),
        "sample_rows": df_sample_rows.to_string(index=False)
    })
    
    print(f"LLM Decision: '{result}'")
    return result if result.lower() != 'none' else None

def parse_llm_json(llm_output_str):
    if not isinstance(llm_output_str, str):
        return None
    start_brace = llm_output_str.find('{')
    start_bracket = llm_output_str.find('[')
    
    if start_brace == -1 and start_bracket == -1:
        return None 
    
    if start_brace == -1:
        start_index = start_bracket
    elif start_bracket == -1:
        start_index = start_brace
    else:
        start_index = min(start_brace, start_bracket)

    if llm_output_str[start_index] == '{':
        end_index = llm_output_str.rfind('}')
    else:
        end_index = llm_output_str.rfind(']')

    if end_index == -1:
        return None

    json_str = llm_output_str[start_index : end_index + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def generate_new_table_details(df_columns):
    """
    Uses an LLM to generate a table name and a detailed column mapping,
    with a retry mechanism for up to 2 retries.
    """
    print("\nStep 3: Asking LLM to generate a new table schema...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert database designer. Your task is to create a schema for a new SQL table based on a list of column headers from a dataframe.

Dataframe Headers: {columns}

Guidelines:
1.  Generate a suitable SQL table name in `snake_case`.
2.  For each dataframe header, generate a corresponding SQL column name (also in `snake_case`) and infer the most appropriate SQL data type (TEXT, REAL, INTEGER).
3.  Return a JSON object with two keys: 'table_name' and 'columns'.
    - 'table_name': The name you generated.
    - 'columns': A LIST of objects, where each object has three keys: 'df_col' (the original dataframe header), 'sql_col' (the generated SQL column name), and 'sql_type' (the SQL data type).
    
Example Response:
{{
  "table_name": "example_products",
  "columns": [
    {{"df_col": "product_name", "sql_col": "product_name", "sql_type": "TEXT"}},
    {{"df_col": "item_price", "sql_col": "price", "sql_type": "REAL"}}
  ]
}}
"""),
        ("human", "Generate the JSON for my new table.")
    ])

    chain = prompt | llm | StrOutputParser()
    
    # --- Retry Mechanism ---
    max_retries = 2
    for attempt in range(max_retries + 1): # Total of 3 attempts
        try:
            print(f"LLM schema generation: Attempt {attempt + 1}/{max_retries + 1}...")
            response_str = chain.invoke({"columns": ", ".join(df_columns)})
            
            # The LLM is instructed to return JSON, so we parse it.
            result = parse_llm_json(response_str)
            print(f"LLM generated schema successfully: {result}")
            return result # Success: exit the function with the result

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error on attempt {attempt + 1}: Could not parse LLM response. Error: {e}")
            if attempt < max_retries:
                print("Retrying...")
            else:
                print("All retry attempts have failed.")
    
    # If all attempts fail after the loop, return None
    return None


def generate_existing_table_column_map(table_schema, df_columns):
    """Uses an LLM to map DataFrame columns to an existing table's columns."""
    print(f"\nStep 3b: Asking LLM to map columns to existing table...")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert database administrator. Your task is to map columns from a new dataset to an existing SQL table.

Here is the schema of the target SQL table:
{table_schema}

Here are the headers from the new dataset:
{df_columns}

Instructions:
- Create a mapping from the new dataset's headers to the SQL table's columns.
- Return a JSON object with a single key: 'columns'.
- 'columns' should be a LIST of objects, where each object has two keys: 'df_col' (the header from the new data) and 'sql_col' (the corresponding column in the SQL table).
- Only include mappings for columns that have a clear semantic match. If a column from the new data doesn't fit, omit it.
"""
    )
    chain = prompt | llm | StrOutputParser()
    response_str = chain.invoke({
        "table_schema": table_schema,
        "df_columns": ", ".join(df_columns)
    })
    try:
        result = parse_llm_json(response_str)
        if result:
            return result['columns']
        else:
            return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing LLM response for column map: {e}")
        return None

def remove_backslash_except_backslash(text: str) -> str:
    return re.sub(r'\\([^\\])', r'\1', text)


def ingest_markdown_table(md_table: str, file_name: str, file_size: int, conn = None) -> FileIngestionResult | TableDetails:
    
    print("=====================================================")
    print("Starting new ingestion process...")
    print("Input Markdown Table:")
    print("=====================================================")

    conn = get_db_connection()

    sql_commands = []

    print("\nStep 1: Retrieving current database schema...")
    schema = get_db_schema(conn)
    if len(schema.strip()) == 0:
        print("No existing tables found. Will create a new table.")

    df = parse_markdown_table(remove_backslash_except_backslash(md_table))

    target_table = get_matching_table_name(
        schema=schema,
        df_columns=df.columns.tolist(),
        df_sample_rows=df.head(2)
    )
    
    column_map = None

    if target_table:
        table_schema = get_table_schema(conn, target_table)
        column_map = generate_existing_table_column_map(table_schema, df.columns.tolist())
    else:
        schema_details = generate_new_table_details(df.columns.tolist())
        if not schema_details:
            print("Could not generate a valid new table schema. Aborting.")
            conn.close()
            raise Exception("LLM failed to generate a valid new table schema.")
        if schema_details and 'table_name' in schema_details and 'columns' in schema_details:
            target_table = schema_details['table_name']
            column_map = schema_details['columns']
            
            cols_sql_parts = [f"\"{col['sql_col']}\" {col['sql_type']}" for col in column_map]
            create_sql = f"CREATE TABLE IF NOT EXISTS {target_table} ({', '.join(cols_sql_parts)})"
            sql_commands.append(create_sql)
            
            print(f"Executing CREATE TABLE statement: {create_sql}")
            execute_query(conn, create_sql)
        else:
            print("Could not generate a valid new table schema. Aborting.")
            conn.close()
            raise Exception("LLM response for new table schema was malformed.")

    if target_table and column_map:
        print(f"\nStep 4: Inserting data into '{target_table}' using explicit column map...")

        sql_cols = [col['sql_col'] for col in column_map]
        df_cols_ordered = [col['df_col'] for col in column_map]
        df_for_insert = df[df_cols_ordered].copy()

        # Get SQL schema for target table
        table_schema = get_table_schema_json(conn, target_table)  # {col_name: sql_type}
        
        # Function to check if value matches SQL type
        def validate_value(value, sql_type):
            if pd.isna(value) or value == "":
                return None
            try:
                if "INT" in sql_type.upper():
                    return int(value)
                elif "REAL" in sql_type.upper() or "FLOAT" in sql_type.upper() or "DOUBLE" in sql_type.upper():
                    return float(value)
                elif "CHAR" in sql_type.upper() or "TEXT" in sql_type.upper() or "CLOB" in sql_type.upper():
                    return str(value)
                elif "DATE" in sql_type.upper():
                    # Add date parsing logic if needed
                    return str(value)
                else:
                    return value
            except (ValueError, TypeError):
                return None  # If conversion fails, set to NULL

        # Validate and clean the dataframe before insert
        for i, sql_col in enumerate(sql_cols):
            sql_type = table_schema[sql_col]
            df_for_insert.iloc[:, i] = df_for_insert.iloc[:, i].apply(lambda v: validate_value(v, sql_type))

        columns_str = ', '.join(f'"{c}"' for c in sql_cols)
        placeholders = ', '.join(['?'] * len(sql_cols))
        insert_sql = f"INSERT INTO {target_table} ({columns_str}) VALUES ({placeholders})"
        sql_commands.append(insert_sql)

        rows_to_insert = [tuple(row) for row in df_for_insert.itertuples(index=False)]

        cursor = conn.cursor()
        cursor.executemany(insert_sql, rows_to_insert)
        conn.commit()
        print(f"✅ Successfully inserted {len(rows_to_insert)} rows.")

        print_table_data(conn, target_table)

        table_details = get_table_schema_json(conn, target_table)
        conn.close()

        schema_details_list = [
            ColumnSchema(name=col['sql_col'], type=table_details[col['sql_col']]) 
            for col in column_map
        ]

        return TableDetails(
            tableName=target_table,
            schema_details=schema_details_list,
            rowsInserted=len(rows_to_insert),
            sqlCommands=sql_commands
        )

    else:
        print("❌ Process failed. No target table or column map could be determined.")
        conn.close()
        raise Exception("Could not determine target table or column map.")
