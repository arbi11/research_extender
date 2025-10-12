#!/usr/bin/env python3
"""
Fix notebook 04 q_learning_fundamentals
"""

def fix_notebook_04():
    """Fix the JSON issue in 04_q_learning_fundamentals.ipynb"""
    file_path = "jupyter_book/Chp4/content/04_q_learning_fundamentals.ipynb"

    print(f"Fixing {file_path}...")

    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Fix line 168 (index 167) - similar issue with docstring
    original_line = lines[167]
    print(f"Original line 168: {repr(original_line)}")

    # Replace the malformed line with a properly formatted one
    lines[167] = '        "        \\"\\"\\"Q-learning update function\\"\\"\\"\\n",\n'

    print(f"Fixed line 168: {repr(lines[167])}")

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("Line 168 has been fixed")

    # Validate the fix
    import json
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"OK: {file_path} - Valid JSON with {len(data['cells'])} cells")
        return True
    except Exception as e:
        print(f"ERROR: {file_path} - {e}")
        return False

if __name__ == "__main__":
    fix_notebook_04()