#!/usr/bin/env python3
"""
Batch AST Extraction Script for Python

Extract ASTs from Python files in a directory structure.

python extract_KG_python.py /path/to/python/project --output-folder pyAST --no-copy-source
python src/graph_generator/extract_KG_python.py \
    thesis_code/Chp2_MagneticFieldPredictor/magnetic_field_predictor \
    --output-folder AST_code/AST_Chp2_FieldDistribution \
    --no-copy-source


python src/graph_generator/extract_KG_python.py thesis_codeChp2_MagneticFieldPredictor/magnetic_field_predictor --output-folder AST_code/AST_Chp2_FieldDistribution --no-copy-source
uv run python extract_KG_python.py ../../thesis_code/Chp2_MagneticFieldPredictor/magnetic_field_predictor --output-folder ../../AST_code/AST_Chp2_FieldDistribution --no-copy-source



"""

import argparse
import os
import shutil
import json
from pathlib import Path
from python_code_extractor import extract_symbols

def extract_asts_with_structure(root_folder, output_folder="pyAST", copy_source=True, source_folder="py_source"):
    root_path = Path(root_folder)
    output_path = Path('.').absolute() / output_folder
    source_copy_path = Path('.').absolute() / source_folder if copy_source else None

    if not root_path.exists():
        print(f"Error: {root_folder} does not exist")
        return

    # Create output directories
    output_path.mkdir(exist_ok=True)
    if copy_source:
        source_copy_path.mkdir(exist_ok=True)

    print(f"Extracting ASTs from: {root_path}")
    print(f"AST output directory: {output_path.absolute()}")
    if copy_source:
        print(f"Source copy directory: {source_copy_path.absolute()}")
    print("-" * 60)

    py_files_found = []
    total_definitions = 0
    total_references = 0

    # Find all Python files
    for py_file in root_path.rglob("*.py"):
        # Calculate relative path from root
        relative_path = py_file.relative_to(root_path)
        py_files_found.append((py_file, relative_path))

    print(f"Found {len(py_files_found)} Python files")
    print("-" * 60)

    # Process each Python file
    for i, (py_file, relative_path) in enumerate(py_files_found, 1):
        print(f"[{i}/{len(py_files_found)}] Processing: {relative_path}")

        # Create corresponding directory structure in both output folders
        output_file_path = output_path / relative_path.with_suffix('.json')
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if copy_source:
            source_file_path = source_copy_path / relative_path
            source_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy original Python file
        if copy_source and py_file != source_file_path:
            shutil.copy2(py_file, source_file_path)
            print(f"   📄 Source copied: {source_file_path}")
        elif copy_source:
            print(f"   📄 Source already exists: {source_file_path}")

        # Read the code from the file
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Use the new Python AST extractor
        result = extract_symbols(code, str(py_file))

        definitions = result.get('definitions', [])
        references = result.get('references', [])
        
        result_with_metadata = {
            "file_path": str(py_file),
            "language": "python",
            "definitions": definitions,
            "references": references
        }

        # Save AST to JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_with_metadata, f, indent=2, ensure_ascii=False)

        print(f"   ✅ AST saved: {output_file_path}")
        print(f"   📊 Definitions: {len(definitions)}, References: {len(references)}")
        total_definitions += len(definitions)
        total_references += len(references)


    print("-" * 60)
    print(f'Total definitions: {total_definitions}, Total references: {total_references}')
    print(f"AST extraction complete! Check {output_path.absolute()}")
    if copy_source:
        print(f"Source files copied! Check {source_copy_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Extract ASTs from Python files.")
    parser.add_argument("root_folder", help="Root folder containing Python files")
    parser.add_argument("--output-folder", default="pyAST", help="Output folder for AST JSON files (default: pyAST)")
    parser.add_argument("--source-folder", default="py_source", help="Folder to copy source files (default: py_source)")
    parser.add_argument("--no-copy-source", action="store_true", help="Don't copy source files")

    args = parser.parse_args()

    copy_source = not args.no_copy_source
    
    root_folder = Path(args.root_folder).absolute()

    extract_asts_with_structure(
        root_folder=root_folder,
        output_folder=args.output_folder,
        copy_source=copy_source,
        source_folder=args.source_folder
    )

if __name__ == "__main__":
    main()
