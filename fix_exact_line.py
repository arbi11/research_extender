#!/usr/bin/env python3
"""
Fix the exact JSON issue at line 314
"""

def fix_line_314():
    """Fix the specific JSON issue at line 314"""
    file_path = "jupyter_book/Chp4/content/03_markov_decision_processes.ipynb"

    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Fix line 314 (index 313)
    original_line = lines[313]
    print(f"Original line 314: {repr(original_line)}")

    # The line should be a proper JSON string containing a Python docstring
    # Replace the malformed line with a properly formatted one
    lines[313] = '        "        \\"\\"\\"Initialize topology with some material\\"\\"\\"\\n",\n'

    print(f"Fixed line 314: {repr(lines[313])}")

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("Line 314 has been fixed")

if __name__ == "__main__":
    fix_line_314()