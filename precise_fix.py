#!/usr/bin/env python3
"""
Precise fix for the docstring issue in MDP notebook
"""

def fix_precise():
    """Fix the exact JSON structure issue"""
    file_path = "jupyter_book/Chp4/content/03_markov_decision_processes.ipynb"

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find and fix line 314 (index 313)
    for i, line in enumerate(lines):
        if '\"\"\"Initialize topology with some material\"\"\"' in line:
            print(f"Found problematic line at index {i+1}: {line.strip()}")
            # Replace it with properly formatted JSON string
            lines[i] = '        "        \\"\\"\\"Initialize topology with some material\\"\\"\\"\\n",\n'
            print(f"Fixed line: {lines[i].strip()}")
            break

    # Write back the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("Fix applied successfully")

if __name__ == "__main__":
    fix_precise()