#!/usr/bin/env python3
"""
Script to fix the specific docstring issue in the MDP notebook
"""

import json
import re

def fix_mdp_notebook():
    """Fix the malformed docstring in 03_markov_decision_processes.ipynb"""
    file_path = "jupyter_book/Chp4/content/03_markov_decision_processes.ipynb"

    print(f"Reading {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the problematic line and fix it
    # The issue is that the docstring is not properly enclosed in JSON string format
    content = content.replace(
        '    def reset(self):\n',
        '    def reset(self):\\n',
    )

    content = content.replace(
        '\"\"\"Initialize topology with some material\"\"\"\n',
        '        \"        \"\"\"Initialize topology with some material\"\"\"\\n',
    )

    try:
        # Try to parse the JSON to verify it's valid
        data = json.loads(content)
        print("JSON is valid after fixing")

        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Successfully fixed {file_path}")
        return True

    except json.JSONDecodeError as e:
        print(f"JSON still has errors: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        return False

if __name__ == "__main__":
    fix_mdp_notebook()