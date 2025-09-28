#!/usr/bin/env python3
"""
LaTeX AST Extraction Script

Extract structured information from LaTeX files using plasTeX.
Emits JSON in the same "definitions + references" format as code ASTs.

Usage:
python latex_code_extractor.py /path/to/latex/files --output-folder AST_code/AST_LaTeX --pattern "*.tex"
"""

import argparse
import os
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# plasTeX imports
from plasTeX.TeX import TeX
from plasTeX.Renderers.Text import Renderer
from plasTeX import Document


def normalize_text(node) -> str:
    """Convert plasTeX node to plain text, handling TeXFragment objects"""
    if hasattr(node, 'textContent'):
        return str(node.textContent).strip()
    elif hasattr(node, 'children'):
        # Recursively extract text from children
        text_parts = []
        for child in node.children:
            text_parts.append(normalize_text(child))
        return ' '.join(text_parts).strip()
    else:
        return str(node).strip()


def extract_latex_structure(latex_file_path: str) -> Dict[str, Any]:
    """
    Extract structured information from LaTeX file using regex patterns.
    Fallback approach when plasTeX fails due to complex LaTeX.
    """
    try:
        # Read the LaTeX file content
        with open(latex_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_stem = Path(latex_file_path).stem
        definitions = []
        references = []

        # Build label mapping for reference resolution
        label_map = {}

        # Extract sections using regex with position tracking
        section_patterns = [
            (r'\\chapter\s*\{([^}]+)\}', 'Chapter', 'chapter'),
            (r'\\section\s*\{([^}]+)\}', 'Section', 'section'),
            (r'\\subsection\s*\{([^}]+)\}', 'Subsection', 'subsection'),
            (r'\\subsubsection\s*\{([^}]+)\}', 'Subsubsection', 'subsubsection'),
        ]

        # Collect all section matches with positions
        section_matches = []
        for pattern, section_type, node_type in section_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                section_matches.append({
                    'match': match,
                    'title': match.group(1).strip(),
                    'type': section_type,
                    'node_type': node_type,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })

        # Sort by position
        section_matches.sort(key=lambda x: x['start_pos'])

        # Compute line/column positions and section ranges
        for i, sec_info in enumerate(section_matches):
            match = sec_info['match']
            title = sec_info['title']
            section_type = sec_info['type']
            node_type = sec_info['node_type']
            start_pos = sec_info['start_pos']

            # Compute start line and column
            start_line = content.count('\n', 0, start_pos) + 1
            last_newline_before = content.rfind('\n', 0, start_pos)
            start_col = start_pos - last_newline_before if last_newline_before >= 0 else start_pos + 1

            # Compute end position (next section of same or higher level, or end of file)
            end_pos = len(content)
            current_level = {'chapter': 0, 'section': 1, 'subsection': 2, 'subsubsection': 3}[node_type]

            for j in range(i + 1, len(section_matches)):
                next_sec = section_matches[j]
                next_level = {'chapter': 0, 'section': 1, 'subsection': 2, 'subsubsection': 3}[next_sec['node_type']]
                if next_level <= current_level:
                    end_pos = next_sec['start_pos']
                    break

            # Compute end line and column
            end_line = content.count('\n', 0, end_pos) + 1
            last_newline_before_end = content.rfind('\n', 0, end_pos)
            end_col = end_pos - last_newline_before_end if last_newline_before_end >= 0 else end_pos + 1

            # Look for label after this section (forward search)
            label_pattern = r'\\label\s*\{([^}]+)\}'
            label_match = re.search(label_pattern, content[start_pos:start_pos+500])  # Look 500 chars ahead
            label = label_match.group(1) if label_match else f"{node_type}_{i+1}"

            hierarchical_name = f"{file_stem}::{title}"

            definition = {
                "name": hierarchical_name,
                "type": section_type,
                "category": "definition",
                "label": label,
                "order": i + 1,
                "start_line": start_line,
                "end_line": end_line,
                "start_col": start_col,
                "end_col": end_col,
                "file_path": latex_file_path
            }
            definitions.append(definition)

            if label:
                label_map[label] = hierarchical_name

        # Extract equations
        equation_patterns = [
            (r'\\begin\s*\{equation\}(.*?)\\end\s*\{equation\}', 'Equation', 'equation'),
            (r'\\begin\s*\{align\}(.*?)\\end\s*\{align\}', 'Equation', 'align'),
            (r'\\begin\s*\{gather\}(.*?)\\end\s*\{gather\}', 'Equation', 'gather'),
            (r'\\begin\s*\{multline\}(.*?)\\end\s*\{multline\}', 'Equation', 'multline'),
            (r'\\begin\s*\{eqnarray\}(.*?)\\end\s*\{eqnarray\}', 'Equation', 'eqnarray'),
        ]

        eq_counter = 1
        for pattern, eq_type, node_type in equation_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                eq_content = match.group(1).strip()
                start_pos = match.start()

                # Look for label before this equation
                label_pattern = r'\\label\s*\{([^}]+)\}'
                label_match = re.search(label_pattern, content[:start_pos])
                eq_label = label_match.group(1) if label_match else f"eq_{eq_counter}"

                definition = {
                    "name": f"{file_stem}::{eq_label}",
                    "type": eq_type,
                    "category": "definition",
                    "label": eq_label,
                    "order": eq_counter,
                    "file_path": latex_file_path
                }
                definitions.append(definition)

                if eq_label:
                    label_map[eq_label] = definition["name"]

                eq_counter += 1

        # Extract figures
        figure_pattern = r'\\begin\s*\{figure\}(.*?)\\end\s*\{figure\}'
        fig_matches = re.findall(figure_pattern, content, re.DOTALL)

        fig_counter = 1
        for fig_content in fig_matches:
            # Look for caption
            caption_pattern = r'\\caption\s*\{([^}]+)\}'
            caption_match = re.search(caption_pattern, fig_content)
            caption = caption_match.group(1).strip() if caption_match else ""

            # Look for label in figure content
            label_pattern = r'\\label\s*\{([^}]+)\}'
            label_match = re.search(label_pattern, fig_content)
            fig_label = label_match.group(1) if label_match else f"fig_{fig_counter}"

            definition = {
                "name": f"{file_stem}::{fig_label}",
                "type": "Figure",
                "category": "definition",
                "label": fig_label,
                "order": fig_counter,
                "file_path": latex_file_path
            }
            definitions.append(definition)

            if fig_label:
                label_map[fig_label] = definition["name"]

            fig_counter += 1

        # Extract tables
        table_pattern = r'\\begin\s*\{table\}(.*?)\\end\s*\{table\}'
        tab_matches = re.findall(table_pattern, content, re.DOTALL)

        tab_counter = 1
        for tab_content in tab_matches:
            # Look for caption
            caption_pattern = r'\\caption\s*\{([^}]+)\}'
            caption_match = re.search(caption_pattern, tab_content)
            caption = caption_match.group(1).strip() if caption_match else ""

            # Look for label in table content
            label_pattern = r'\\label\s*\{([^}]+)\}'
            label_match = re.search(label_pattern, tab_content)
            tab_label = label_match.group(1) if label_match else f"tab_{tab_counter}"

            definition = {
                "name": f"{file_stem}::{tab_label}",
                "type": "Table",
                "category": "definition",
                "label": tab_label,
                "order": tab_counter,
                "file_path": latex_file_path
            }
            definitions.append(definition)

            if tab_label:
                label_map[tab_label] = definition["name"]

            tab_counter += 1

        # Extract algorithms
        alg_pattern = r'\\begin\s*\{algorithm\}(.*?)\\end\s*\{algorithm\}'
        alg_matches = re.findall(alg_pattern, content, re.DOTALL)

        alg_counter = 1
        for alg_content in alg_matches:
            # Look for caption
            caption_pattern = r'\\caption\s*\{([^}]+)\}'
            caption_match = re.search(caption_pattern, alg_content)
            caption = caption_match.group(1).strip() if caption_match else ""

            # Look for label in algorithm content
            label_pattern = r'\\label\s*\{([^}]+)\}'
            label_match = re.search(label_pattern, alg_content)
            alg_label = label_match.group(1) if label_match else f"alg_{alg_counter}"

            definition = {
                "name": f"{file_stem}::{alg_label}",
                "type": "Algorithm",
                "category": "definition",
                "label": alg_label,
                "order": alg_counter,
                "file_path": latex_file_path
            }
            definitions.append(definition)

            if alg_label:
                label_map[alg_label] = definition["name"]

            alg_counter += 1

        # Extract theorem-like environments
        theorem_patterns = [
            (r'\\begin\s*\{theorem\}(.*?)\\end\s*\{theorem\}', 'Theorem'),
            (r'\\begin\s*\{lemma\}(.*?)\\end\s*\{lemma\}', 'Lemma'),
            (r'\\begin\s*\{proposition\}(.*?)\\end\s*\{proposition\}', 'Proposition'),
            (r'\\begin\s*\{corollary\}(.*?)\\end\s*\{corollary\}', 'Corollary'),
            (r'\\begin\s*\{definition\}(.*?)\\end\s*\{definition\}', 'Definition'),
            (r'\\begin\s*\{remark\}(.*?)\\end\s*\{remark\}', 'Remark'),
        ]

        theorem_counter = 1
        for pattern, theorem_type in theorem_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for theorem_content in matches:
                # Look for label in theorem content
                label_pattern = r'\\label\s*\{([^}]+)\}'
                label_match = re.search(label_pattern, theorem_content)
                theorem_label = label_match.group(1) if label_match else f"{theorem_type.lower()}_{theorem_counter}"

                definition = {
                    "name": f"{file_stem}::{theorem_label}",
                    "type": theorem_type,
                    "category": "definition",
                    "label": theorem_label,
                    "order": theorem_counter,
                    "file_path": latex_file_path
                }
                definitions.append(definition)

                if theorem_label:
                    label_map[theorem_label] = definition["name"]

                theorem_counter += 1

        # Extract references from the entire document
        all_refs = extract_refs_from_text(content, file_stem, latex_file_path)
        references.extend(all_refs)

        # Remove duplicate references
        seen_refs = set()
        unique_references = []
        for ref in references:
            ref_key = (ref['name'], ref['type'], ref['calling_entity'])
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                unique_references.append(ref)

        references = unique_references

        return {
            "file_path": latex_file_path,
            "language": "latex",
            "definitions": definitions,
            "references": references
        }

    except Exception as e:
        print(f"Error processing {latex_file_path}: {e}")
        return {
            "file_path": latex_file_path,
            "language": "latex",
            "definitions": [],
            "references": []
        }


def extract_refs_from_text(text: str, calling_entity: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract LaTeX references from text"""
    references = []

    # Pattern for \ref{label}
    ref_pattern = r'\\ref\s*\{([^}]+)\}'
    for match in re.finditer(ref_pattern, text):
        label = match.group(1)
        references.append({
            "name": label,
            "type": "ref",
            "category": "reference",
            "calling_entity": calling_entity,
            "file_path": file_path
        })

    # Pattern for \eqref{label}
    eqref_pattern = r'\\eqref\s*\{([^}]+)\}'
    for match in re.finditer(eqref_pattern, text):
        label = match.group(1)
        references.append({
            "name": label,
            "type": "eqref",
            "category": "reference",
            "calling_entity": calling_entity,
            "file_path": file_path
        })

    # Pattern for \autoref{label}
    autoref_pattern = r'\\autoref\s*\{([^}]+)\}'
    for match in re.finditer(autoref_pattern, text):
        label = match.group(1)
        references.append({
            "name": label,
            "type": "autoref",
            "category": "reference",
            "calling_entity": calling_entity,
            "file_path": file_path
        })

    # Pattern for \cite{key1,key2}
    cite_pattern = r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}'
    for match in re.finditer(cite_pattern, text):
        keys = match.group(1)
        # Split multiple keys
        for key in keys.split(','):
            key = key.strip()
            if key:
                references.append({
                    "name": key,
                    "type": "cite",
                    "category": "reference",
                    "calling_entity": calling_entity,
                    "file_path": file_path
                })

    return references


def extract_latex_files(root_folder: str, output_folder: str = "AST_code/AST_LaTeX",
                       pattern: str = "*.tex", copy_source: bool = True,
                       source_folder: str = "latex_source") -> None:
    """Extract LaTeX structure from all .tex files in directory"""

    root_path = Path(root_folder)
    output_path = Path(output_folder)
    source_copy_path = Path(source_folder) if copy_source else None

    if not root_path.exists():
        print(f"Error: {root_folder} does not exist")
        return

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    if copy_source:
        source_copy_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting LaTeX ASTs from: {root_path}")
    print(f"AST output directory: {output_path.absolute()}")
    if copy_source:
        print(f"Source copy directory: {source_copy_path.absolute()}")
    print("-" * 60)

    tex_files_found = []
    total_definitions = 0
    total_references = 0

    # Find all LaTeX files
    for tex_file in root_path.rglob(pattern):
        # Calculate relative path from root
        relative_path = tex_file.relative_to(root_path)
        tex_files_found.append((tex_file, relative_path))

    print(f"Found {len(tex_files_found)} LaTeX files")
    print("-" * 60)

    # Process each LaTeX file
    for i, (tex_file, relative_path) in enumerate(tex_files_found, 1):
        print(f"[{i}/{len(tex_files_found)}] Processing: {relative_path}")

        # Create corresponding directory structure in both output folders
        output_file_path = output_path / relative_path.with_suffix('.json')
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if copy_source:
            source_file_path = source_copy_path / relative_path
            source_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy original LaTeX file
        if copy_source and tex_file != source_file_path:
            shutil.copy2(tex_file, source_file_path)
            print(f"   📄 Source copied: {source_file_path}")
        elif copy_source:
            print(f"   📄 Source already exists: {source_file_path}")

        # Extract LaTeX structure
        result = extract_latex_structure(str(tex_file))

        definitions = result.get('definitions', [])
        references = result.get('references', [])

        # Save AST to JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"   ✅ AST saved: {output_file_path}")
        print(f"   📊 Definitions: {len(definitions)}, References: {len(references)}")
        total_definitions += len(definitions)
        total_references += len(references)

    print("-" * 60)
    print(f'Total definitions: {total_definitions}, Total references: {total_references}')
    print(f"LaTeX AST extraction complete! Check {output_path.absolute()}")
    if copy_source:
        print(f"Source files copied! Check {source_copy_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Extract AST-like structure from LaTeX files.")
    parser.add_argument("root_folder", help="Root folder containing LaTeX files")
    parser.add_argument("--output-folder", default="AST_code/AST_LaTeX",
                       help="Output folder for AST JSON files (default: AST_code/AST_LaTeX)")
    parser.add_argument("--pattern", default="*.tex",
                       help="File pattern to match (default: *.tex)")
    parser.add_argument("--source-folder", default="latex_source",
                       help="Folder to copy source files (default: latex_source)")
    parser.add_argument("--no-copy-source", action="store_true",
                       help="Don't copy source files")

    args = parser.parse_args()

    copy_source = not args.no_copy_source

    extract_latex_files(
        root_folder=args.root_folder,
        output_folder=args.output_folder,
        pattern=args.pattern,
        copy_source=copy_source,
        source_folder=args.source_folder
    )


if __name__ == "__main__":
    main()
