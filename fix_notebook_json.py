#!/usr/bin/env python3
"""
Script to fix corrupted JSON in notebook files
"""

import json
import re
from pathlib import Path

def fix_notebook_json(file_path):
    """Fix corrupted JSON in notebook file"""
    print(f"Fixing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the start and end of JSON
    if not content.strip().startswith('{'):
        print(f"Error: File doesn't start with JSON object")
        return False

    # Try to find complete JSON structure
    brace_count = 0
    json_end = -1

    for i, char in enumerate(content):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break

    if json_end == -1:
        print(f"Error: Could not find complete JSON structure")
        return False

    # Extract JSON content
    json_content = content[:json_end]

    # Fix common JSON issues
    # Remove Python-style comments
    json_content = re.sub(r'#.*$', '', json_content, flags=re.MULTILINE)

    # Fix trailing commas in arrays/objects
    json_content = re.sub(r',\s*([}\]])', r'\1', json_content)

    # Fix malformed string literals
    json_content = re.sub(r'""\s*\+\s*"', '"', json_content)

    try:
        # Try to parse the JSON
        data = json.loads(json_content)
        print(f"Successfully parsed JSON")

        # Write back the fixed JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Fixed {file_path}")
        return True

    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        print(f"Error location: line {e.lineno}, column {e.colno}")
        return False

def main():
    """Main function"""
    notebook_files = [
        "jupyter_book/Chp4/content/02_literature_survey.ipynb",
        "jupyter_book/Chp4/content/03_markov_decision_processes.ipynb"
    ]

    for notebook_file in notebook_files:
        fix_notebook_json(notebook_file)

if __name__ == "__main__":
    main()