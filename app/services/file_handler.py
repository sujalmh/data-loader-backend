import os
import re
from pathlib import Path
from typing import List

class MarkdownTableExtractor:
    """Service to extract tables from markdown files and save them separately."""

    def extract_tables_from_file(self, markdown_file_path: str, output_folder: str = None) -> List[str]:
        """
        Extract all tables from a markdown file and save them as separate files.

        Args:
            markdown_file_path: Path to the source markdown file.
            output_folder: Folder to save extracted tables.
                           Defaults to a new folder next to the source file.

        Returns:
            A list of paths to the created table files.
        """
        source_path = Path(markdown_file_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"File not found: {markdown_file_path}")

        with source_path.open('r', encoding='utf-8') as file:
            content = file.read()

        if output_folder is None:
            output_folder = source_path.parent / f"{source_path.stem}_tables"
        
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)

        tables = self._find_tables(content)
        created_files = []

        for i, table_content in enumerate(tables, 1):
            table_filename = f"table_{i:02d}.md"
            table_filepath = output_path / table_filename
            
            with table_filepath.open('w', encoding='utf-8') as table_file:
                table_file.write(table_content.strip() + '\n')
            
            created_files.append(str(table_filepath))
        
        return created_files

    def _find_tables(self, content: str) -> List[str]:
        """Find all table blocks in markdown content."""
        tables = []
        lines = content.split('\n')
        current_table_lines = []
        in_table = False

        for line in lines:
            if self._is_table_row(line):
                if not in_table:
                    in_table = True
                current_table_lines.append(line)
            else:
                if in_table:
                    if self._is_valid_table(current_table_lines):
                        tables.append('\n'.join(current_table_lines))
                    current_table_lines = []
                    in_table = False
        
        if in_table and self._is_valid_table(current_table_lines):
            tables.append('\n'.join(current_table_lines))

        return tables

    def _is_table_row(self, line: str) -> bool:
        """Check if a line appears to be part of a markdown table."""
        stripped = line.strip()
        return stripped.startswith('|') and stripped.endswith('|')

    def _is_valid_table(self, table_lines: List[str]) -> bool:
        """Validate that the lines form a proper table with a header separator."""
        if len(table_lines) < 2:
            return False
        
        # The second line must be a separator line, e.g., |---|---|
        separator_line = table_lines[1].strip()
        return bool(re.match(r'^\|(?:\s*:?-+:?\s*\|)+$', separator_line))

def extract_markdown_tables(file_path: str, output_folder: str = None) -> List[str]:
    """
    Convenience function to extract tables from a markdown file.

    Args:
        file_path: Path to the markdown file.
        output_folder: Optional path to the output folder.

    Returns:
        A list of paths to the created table files.
    """
    extractor = MarkdownTableExtractor()
    return extractor.extract_tables_from_file(file_path, output_folder)

import re

def is_separator_row(line: str) -> bool:
    parts = line.split('|')
    if parts and parts[0].strip() == '':
        parts = parts[1:]
    if parts and parts[-1].strip() == '':
        parts = parts[:-1]
    if len(parts) == 0:
        return False
    pattern = re.compile(r'^\s*:?-+:?\s*$')
    for part in parts:
        if not pattern.match(part):
            return False
    return True

def is_separator_cells(cells: list) -> bool:
    pattern = re.compile(r'^\s*:?-+:?\s*$')
    for cell in cells:
        if not pattern.match(cell):
            return False
    return len(cells) > 0

def fix_markdown_tables(text: str) -> str:
    lines = text.splitlines()
    blocks = []
    start = None
    for i, line in enumerate(lines):
        if '|' in line:
            if start is None:
                start = i
        else:
            if start is not None:
                blocks.append((start, i-1))
                start = None
    if start is not None:
        blocks.append((start, len(lines)-1))
    
    new_lines = []
    last_end = -1
    for (start, end) in blocks:
        if last_end + 1 <= start - 1:
            new_lines.extend(lines[last_end+1:start])
        
        table_block = lines[start:end+1]
        n_pipes = None
        separator_index_in_block = None
        for idx, line in enumerate(table_block):
            if is_separator_row(line):
                count = line.count('|')
                if n_pipes is None:
                    n_pipes = count
                separator_index_in_block = idx
                break
        
        if n_pipes is None:
            if table_block:
                n_pipes = table_block[0].count('|')
            else:
                n_pipes = 0
        
        merged_rows = []
        current = []
        for line in table_block:
            current.append(line.strip())
            merged_line = ' '.join(current)
            total_pipes = merged_line.count('|')
            if total_pipes == n_pipes:
                merged_rows.append(merged_line)
                current = []
        if current:
            merged_rows.append(' '.join(current))
        
        table_data = []
        for row in merged_rows:
            cells = [cell.strip() for cell in row.split('|')]
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]
            expected_cells = n_pipes - 1
            if len(cells) < expected_cells:
                cells += [''] * (expected_cells - len(cells))
            elif len(cells) > expected_cells:
                cells = cells[:expected_cells]
            table_data.append(cells)
        
        separator_row_idx = None
        for idx, cells in enumerate(table_data):
            if is_separator_cells(cells):
                separator_row_idx = idx
                break
        
        if separator_row_idx is not None and separator_row_idx > 0:
            header_rows = table_data[:separator_row_idx]
            n_cols = len(header_rows[0]) if header_rows else 0
            combined_header = []
            for j in range(n_cols):
                col_data = []
                for i in range(len(header_rows)):
                    if j < len(header_rows[i]):
                        col_data.append(header_rows[i][j])
                    else:
                        col_data.append('')
                combined_header.append(" ".join(col_data).strip())
            body_rows = table_data[separator_row_idx+1:]
        else:
            if table_data:
                combined_header = table_data[0]
                body_rows = table_data[1:]
            else:
                combined_header = []
                body_rows = []
        
        table_output_lines = []
        if combined_header:
            table_output_lines.append("| " + " | ".join(combined_header) + " |")
            table_output_lines.append("| " + " | ".join(['---'] * len(combined_header)) + " |")
            for row in body_rows:
                table_output_lines.append("| " + " | ".join(row) + " |")
        else:
            for row in table_data:
                table_output_lines.append("| " + " | ".join(row) + " |")
        
        new_lines.extend(table_output_lines)
        last_end = end
    
    if last_end < len(lines)-1:
        new_lines.extend(lines[last_end+1:])
    
    return "\n".join(new_lines)

def parse_table(table_text):
    """Convert a markdown table to list of rows (list of cells)."""
    rows = []
    for line in table_text.strip().split("\n"):
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if not all(c == "" or re.match(r"^-+$", c) for c in cells):  # ignore separator
                rows.append(cells)
    return rows

def detect_type(cell):
    """Return 'num', 'str', or 'null'."""
    cell = cell.strip()
    if cell == "" or cell.lower() in ["-", "na", "null"]:
        return "null"
    try:
        float(cell.replace(",", ""))  # allow numbers with commas
        return "num"
    except ValueError:
        return "str"

def parse_row(line):
    """Parse markdown table row into list of cells."""
    return [c.strip() for c in line.strip().split("|")[1:-1]]

def are_rows_compatible(row1, row2):
    """Check if two rows match in types (null ignored)."""
    if len(row1) != len(row2):
        return False
    for c1, c2 in zip(row1, row2):
        t1, t2 = detect_type(c1), detect_type(c2)
        if "null" in (t1, t2):
            continue
        if t1 != t2:
            return False
    return True

def merge_if_similar(md_text):
    lines = md_text.splitlines()
    output_lines = []
    i = 0
    while i < len(lines):
        output_lines.append(lines[i])
        # Check if current line is last of a table
        if lines[i].strip().startswith("|") and (i+2 < len(lines)):
            # Possible blank line between tables
            if lines[i+1].strip() == "" and lines[i+2].strip().startswith("|"):
                last_row = parse_row(lines[i])
                first_row = parse_row(lines[i+2])
                if len(last_row) == len(first_row) and are_rows_compatible(last_row, first_row):
                    # Skip the blank line
                    i += 1
        i += 1
    return "\n".join(output_lines)

def clean_markdown(md_text):
    fixed_text = fix_markdown_tables(md_text)
    merged_markdown = merge_if_similar(fixed_text)
    return merged_markdown