"""
Validation module for input validation.

Pure functions for validating inputs. Easy to test and reason about.
"""

from pathlib import Path
from typing import List, Tuple


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


def validate_task_id(task_id: int) -> bool:
    """
    Validate that task ID is 1 or 2.
    
    Args:
        task_id: The task ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return task_id in (1, 2)


def validate_pdf_file(filepath: Path) -> bool:
    """
    Check if file exists and has PDF extension.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        True if file exists and is a PDF, False otherwise
    """
    if not filepath.exists():
        return False
    
    if not filepath.is_file():
        return False
    
    # Check extension (case-insensitive)
    return filepath.suffix.lower() == '.pdf'


def validate_sources_exist(sources: List[str], base_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate that all source files exist.
    
    Args:
        sources: List of source filenames
        base_dir: Base directory where sources should be located
        
    Returns:
        Tuple of (is_valid, missing_files)
        - is_valid: True if all files exist, False otherwise
        - missing_files: List of filenames that are missing
    """
    missing_files: List[str] = []
    
    for source_file in sources:
        filepath = base_dir / source_file
        if not filepath.exists():
            missing_files.append(source_file)
    
    is_valid = len(missing_files) == 0
    return (is_valid, missing_files)


def validate_output_directory(path: Path) -> bool:
    """
    Ensure output directory exists or can be created.
    
    Args:
        path: Path to the output directory
        
    Returns:
        True if directory exists or can be created, False otherwise
    """
    try:
        if path.exists():
            if not path.is_dir():
                return False
            # Check if we can write to it
            test_file = path / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
                return True
            except (OSError, PermissionError):
                return False
        else:
            # Try to create it
            path.mkdir(parents=True, exist_ok=True)
            return True
    except (OSError, PermissionError):
        return False
