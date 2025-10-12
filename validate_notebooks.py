#!/usr/bin/env python3
"""
Validate JSON structure of notebook files
"""

import json
import sys

def validate_notebook(file_path):
    """Validate a notebook file's JSON structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check basic notebook structure
        if 'cells' not in data:
            print(f"ERROR: {file_path} - missing 'cells' key")
            return False

        if 'metadata' not in data:
            print(f"ERROR: {file_path} - missing 'metadata' key")
            return False

        if 'nbformat' not in data:
            print(f"ERROR: {file_path} - missing 'nbformat' key")
            return False

        print(f"OK {file_path} - Valid JSON structure")
        print(f"  - Contains {len(data['cells'])} cells")
        print(f"  - Notebook format: v{data['nbformat']}.{data.get('nbformat_minor', 0)}")
        return True

    except json.JSONDecodeError as e:
        print(f"ERROR {file_path} - JSON Error: {e}")
        print(f"  - Error location: line {e.lineno}, column {e.colno}")
        return False
    except Exception as e:
        print(f"ERROR {file_path} - Error: {e}")
        return False

def main():
    """Validate all notebooks"""
    notebooks = [
        "jupyter_book/Chp4/content/02_literature_survey.ipynb",
        "jupyter_book/Chp4/content/03_markov_decision_processes.ipynb"
    ]

    print("Validating notebook JSON structures...")
    print("=" * 50)

    all_valid = True
    for notebook in notebooks:
        if not validate_notebook(notebook):
            all_valid = False

    print("=" * 50)
    if all_valid:
        print("OK All notebooks have valid JSON structure!")
    else:
        print("ERROR Some notebooks have JSON errors that need fixing")
        sys.exit(1)

if __name__ == "__main__":
    main()