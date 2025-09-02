import pandas as pd
import numpy as np
import re
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional

# LangChain and Pydantic imports for structured LLM output
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

class ColumnDimension(BaseModel):
    """Defines new dimension columns to be added based on header information."""
    region: Optional[str] = Field(None, description="Geographical region, e.g., 'Urban', 'Rural', 'Combined'")
    state: Optional[str] = Field(None, description="State or province, e.g., 'Delhi'")
    month: Optional[str] = Field(None, description="Month name, e.g., 'June'")
    year: Optional[int] = Field(None, description="Year, e.g., 2024")
    status: Optional[str] = Field(None, description="Status of the data, e.g., 'Final', 'Prov.' (Provisional)")


class ColumnMapping(BaseModel):
    """Defines the transformation for a single original column."""
    new_column: str = Field(..., description="The new, clean name for the column metric, e.g., 'Index', 'Value', 'Weights'.")
    dimensions: Optional[ColumnDimension] = Field(None, description="Additional dimensions extracted from the header.")

class TransformationPlan(BaseModel):
    """The complete transformation plan, mapping all original columns to their new structure."""
    plan: Dict[str, ColumnMapping] = Field(..., description="A dictionary where keys are original column names and values are their transformation mappings.")

def parse_markdown_table(md_str: str) -> pd.DataFrame:
    """
    Parses a markdown table string into a Pandas DataFrame.
    This function is designed to handle tables with multiple header lines,
    intelligently combining them to handle merged/spanned headers. It filters
    out non-semantic header rows (e.g., '(1)', '(2)').

    Args:
        md_str: A string containing the markdown table.

    Returns:
        A Pandas DataFrame representing the initial, potentially untidy, table.
    
    Raises:
        ValueError: If the markdown table format is invalid.
    """
    lines = [line.strip() for line in md_str.strip().split('\n') if line.strip()]
    
    separator_index = -1
    for i, line in enumerate(lines):
        if re.match(r'^[|: -]+$', line):
            separator_index = i
            break
            
    if separator_index == -1:
        raise ValueError("Markdown table separator '---|---' not found.")

    header_lines = lines[:separator_index]
    data_lines = lines[separator_index + 1:]

    # A more robust way to parse cells, avoiding empty strings from start/end pipes
    header_rows = [[cell.strip() for cell in line.split('|')[1:-1]] for line in header_lines]
    data = [[cell.strip() for cell in line.split('|')[1:-1]] for line in data_lines]
    
    # Filter out non-semantic header rows (e.g., those containing only '(1)', '(2)', etc.)
    semantic_header_rows = []
    for row in header_rows:
        is_semantic = any(not re.fullmatch(r'\(\d+\)', cell) for cell in row if cell)
        if is_semantic:
            semantic_header_rows.append(row)
    header_rows = semantic_header_rows

    columns = []
    if not header_rows:
        raise ValueError("No semantic header rows found in Markdown table.")
    elif len(header_rows) == 1:
        columns = header_rows[0]
    else:
        # Handle multi-level (merged) headers by combining them
        num_cols = max(len(row) for row in header_rows)
        for row in header_rows:
            if len(row) < num_cols:
                row.extend([''] * (num_cols - len(row)))
        
        # Start with the bottom-most header as the base
        combined_headers = list(header_rows[-1])
        
        # Prepend prefixes from the rows above
        for i in range(len(header_rows) - 2, -1, -1):
            prefix_row = header_rows[i]
            current_prefix = ""
            for j, cell in enumerate(prefix_row):
                if cell:
                    current_prefix = cell
                # Prepend the prefix if it exists and the current header is not empty
                if current_prefix and combined_headers[j]:
                    combined_headers[j] = f"{current_prefix}_{combined_headers[j]}"
        columns = combined_headers

    if data and len(columns) != len(data[0]):
         raise ValueError(
            f"Column count mismatch. Header has {len(columns)} columns, "
            f"but data row has {len(data[0])} ({data[0]}) cells."
        )

    df = pd.DataFrame(data, columns=columns)
    
    # Replace common non-numeric placeholders with NaN
    df.replace('--', np.nan, inplace=True)

    # Attempt to convert data to numeric types where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass # If conversion fails, leave the column as is (object type)
        
    return df

def normalize_table(df: pd.DataFrame, llm_chain: Any) -> pd.DataFrame:
    """
    Normalizes a DataFrame by unpivoting and merging metrics that have different
    sets of dimensions (e.g., 'Weights' by region and 'Index' by region/month).

    Args:
        df: The initial DataFrame from `parse_markdown_table`.
        llm_chain: The LangChain chain that will generate the transformation plan.

    Returns:
        A tidy Pandas DataFrame.
    """
    original_columns = df.columns.tolist()
    response = llm_chain.invoke({"columns": json.dumps(original_columns)})
    plan = response.plan

    # Identify identifier columns that don't need to be unpivoted
    id_vars = [
        col for col, mapping in plan.items()
        if not mapping.dimensions or not mapping.dimensions.dict(exclude_none=True)
    ]
    
    # Clean up names of id_vars based on the plan
    id_rename_map = {col: plan[col].new_column for col in id_vars}
    df.rename(columns=id_rename_map, inplace=True)
    final_id_vars = list(id_rename_map.values())

    # Group original columns by their new metric name (e.g., 'Index', 'Weights')
    metrics_to_process = defaultdict(list)
    for orig_col, mapping in plan.items():
        if orig_col not in id_vars:
            metrics_to_process[mapping.new_column].append(orig_col)

    metric_dfs = {}
    
    # Process each metric group separately to handle different dimensionalities
    for metric_name, orig_cols in metrics_to_process.items():
        # Melt the dataframe for just the current metric's columns
        melted_df = df.melt(
            id_vars=final_id_vars,
            value_vars=orig_cols,
            var_name='original_column',
            value_name=metric_name
        )
        melted_df.dropna(subset=[metric_name], inplace=True)
        if melted_df.empty:
            continue

        # Create a map of original columns to their dimensions
        dim_map = {
            orig_col: plan[orig_col].dimensions.dict(exclude_none=True)
            for orig_col in orig_cols
        }
        
        # Get all unique dimension keys for this metric group
        all_dims = set(d for dims in dim_map.values() for d in dims.keys())
        for dim in all_dims:
            melted_df[dim] = melted_df['original_column'].apply(
                lambda x: dim_map.get(x, {}).get(dim)
            )

        melted_df.drop(columns=['original_column'], inplace=True)
        
        # Remove rows where none of the relevant dimensions were populated
        if all_dims:
            melted_df.dropna(subset=list(all_dims), how='all', inplace=True)
        
        metric_dfs[metric_name] = melted_df

    if not metric_dfs:
        return df[final_id_vars]

    # Iteratively merge the tidy metric DataFrames, starting with the most granular one
    try:
        base_metric = max(metric_dfs.keys(), key=lambda k: metric_dfs[k].shape[1])
        final_df = metric_dfs.pop(base_metric)
    except ValueError:
        return pd.DataFrame() # No metrics to process

    for metric_name, other_df in metric_dfs.items():
        # Determine common columns to merge on (identifiers + shared dimensions)
        merge_cols = [col for col in other_df.columns if col in final_df.columns and col != metric_name]
        
        final_df = pd.merge(final_df, other_df, on=merge_cols, how='left')

    # Reorder columns for final output
    dim_cols = sorted([col for col in final_df.columns if col not in final_id_vars and col not in metrics_to_process.keys()])
    metric_cols = sorted(list(metrics_to_process.keys()))
    
    final_cols_order = final_id_vars + dim_cols + metric_cols
    # Filter to ensure all columns exist in the final DataFrame
    final_cols_order = [c for c in final_cols_order if c in final_df.columns]
    
    return final_df[final_cols_order]


def run_normalization(md_str: str) -> pd.DataFrame:
    """
    A high-level function that runs the full normalization pipeline.

    Args:
        md_str: A string containing the markdown table.

    Returns:
        A final, tidy Pandas DataFrame.
    """
    print("--- Starting Normalization ---")
    
    # Step 1: Initialize LLM and create the LangChain chain
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, convert_system_message_to_human=True)
    except Exception as e:
        print("\nERROR: Could not initialize Gemini.")
        print("Please ensure the GOOGLE_API_KEY environment variable is set correctly.")
        print(f"Details: {e}")
        return pd.DataFrame() # Return empty df on error

    parser = PydanticOutputParser(pydantic_object=TransformationPlan)
    
    prompt_template = """
    You are an expert data scientist specializing in tidying economic data tables.
    Your task is to analyze the column headers of a DataFrame and create a transformation plan to convert it into a tidy format.

    A tidy format has one observation per row, with variables in columns. The headers you receive are sometimes "denormalized," meaning they contain multiple pieces of information. Your job is to extract this information into new dimension columns.

    RULES:
    - Keep original metric names as final column headers (e.g., 'Index', 'IIP', 'Value', 'Weights'). Do NOT create a generic 'metric' column.
    - Only add new dimension columns if information is present in the headers. Supported dimensions are: 'region', 'state', 'month', 'year', 'status'.
    - Identifier columns without extra dimensions (like 'Date', 'Commodity', 'Year', 'Sl. No.', 'Name of the State/UT') should be mapped to themselves without new dimensions.
    - For status, use 'Final' or 'Prov.' (for Provisional) if present.

    EXAMPLES:
    - Header 'march_23_index' -> new column 'Index', dimensions: {{'month': 'March', 'year': 2023}}.
    - Header 'Delhi_IIP' -> new column 'IIP', dimensions: {{'state': 'Delhi'}}.
    - Header 'Urban_Value' -> new column 'Value', dimensions: {{'region': 'Urban'}}.
    - Header 'Rural_June 24 Index (Final)' -> new column 'Index', dimensions: {{'region': 'Rural', 'month': 'June', 'year': 2024, 'status': 'Final'}}.
    - Header 'Combined_Weights' -> new column 'Weights', dimensions: {{'region': 'Combined'}}.
    - Header 'Year' -> new column 'Year', no extra dimensions.

    Here are the column headers from the table to be transformed:
    {columns}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["columns"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    # Step 2: Parse the raw markdown into an initial DataFrame
    print("1. Parsing markdown table...")
    initial_df = parse_markdown_table(md_str)
    print("Initial DataFrame:")
    print(initial_df)
    print("\nColumns to normalize:", initial_df.columns.tolist())
    
    # Step 3: Use the LLM chain to normalize the DataFrame
    print("\n2. Calling LLM to generate transformation plan...")
    normalized_df = normalize_table(initial_df, chain)
    print("3. Transformation complete.")
    print(normalized_df.head(70))
    print("--- Normalization Finished ---")
    return normalized_df

