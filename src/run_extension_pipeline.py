#!/usr/bin/env python3
"""
Complete pipeline runner for LaTeX + Code → Knowledge Graph → Extensions
Usage: python run_extension_pipeline.py --latex paper.tex --code src/ --output ./extensions
"""

import asyncio
import argparse
from pathlib import Path
import json
import sys

from latex_processor import process_latex
from knowledge_graph_builder import KnowledgeGraphBuilder
from research_extender import ResearchExtender


def collect_code_files(code_paths):
    """Recursively collect all code files from given paths"""
    all_files = []

    for path_str in code_paths:
        path = Path(path_str)

        if path.is_file():
            all_files.append(str(path))
        elif path.is_dir():
            # Collect Python, C++, JavaScript files
            extensions = ['*.py', '*.cpp', '*.c', '*.h', '*.js', '*.ts', '*.java']
            for ext in extensions:
                all_files.extend([str(f) for f in path.rglob(ext)])

    return all_files


async def run_pipeline(latex_file, code_paths, output_dir, graph_dir):
    """Run the complete extension generation pipeline"""

    print(f"Starting Research Extension Pipeline")
    print(f"LaTeX file: {latex_file}")
    print(f"Code paths: {code_paths}")
    print(f"Output directory: {output_dir}")
    print(f"Graph directory: {graph_dir}")

    # Collect all code files
    print("\nCollecting code files...")
    code_files = collect_code_files(code_paths)
    print(f"Found {len(code_files)} code files")

    # Process LaTeX file
    print("\nProcessing LaTeX file...")
    latex_data = process_latex(latex_file)
    print(f"Extracted {len(latex_data['sections'])} sections")
    print(f"Extracted {len(latex_data['equations'])} equations")
    print(f"Extracted {len(latex_data['citations'])} citations")

    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    builder = KnowledgeGraphBuilder(working_dir=graph_dir)
    await builder.build_graph(latex_data, code_files)
    print("Knowledge graph built successfully")

    # Create research extender
    print("\nInitializing research extender...")
    extender = ResearchExtender(builder)

    # Generate extensions
    print("\nGenerating extensions...")
    result = await extender.generate_extensions(latex_file, code_files, output_dir)

    # Print results
    print(f"\n{'='*50}")
    print(f"EXTENSION GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total gaps found: {result['summary']['gaps_found']}")
    print(f"  - Missing implementations: {result['summary']['gaps_by_type']['missing_implementation']}")
    print(f"  - Missing algorithms: {result['summary']['gaps_by_type']['missing_algorithm']}")
    print(f"  - Orphaned code: {result['summary']['gaps_by_type']['orphaned_code']}")

    print(f"\nGenerated {len(result['code_files'])} code files:")
    for filename, filepath in result['code_files'].items():
        print(f"  • {filename} -> {filepath}")

    print(f"\nGenerated {len(result['latex_files'])} LaTeX files:")
    for filename, filepath in result['latex_files'].items():
        print(f"  • {filename} -> {filepath}")

    print(f"\nSummary report: {output_dir}/extension_summary_{result['summary']['timestamp']}.json")
    print(f"Knowledge graph stored in: {graph_dir}")

    return result


def main():
    """Command-line interface"""

    parser = argparse.ArgumentParser(
        description="Generate research extensions from LaTeX papers and code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_extension_pipeline.py --latex paper.tex --code src/ --output ./extensions
  python run_extension_pipeline.py --latex paper.tex --code src1/ src2/ file.py --output ./ext --graph ./kg
        """
    )

    parser.add_argument(
        '--latex',
        required=True,
        help='Path to LaTeX paper file (.tex)'
    )

    parser.add_argument(
        '--code',
        nargs='+',
        required=True,
        help='Paths to code files or directories'
    )

    parser.add_argument(
        '--output',
        default='./research_extensions',
        help='Output directory for generated extensions (default: ./research_extensions)'
    )

    parser.add_argument(
        '--graph',
        default='./research_knowledge_graph',
        help='Directory for knowledge graph storage (default: ./research_knowledge_graph)'
    )

    args = parser.parse_args()

    # Validate inputs
    latex_path = Path(args.latex)
    if not latex_path.exists():
        print(f"Error: LaTeX file not found: {args.latex}")
        sys.exit(1)

    # Create output directories
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.graph).mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        result = asyncio.run(run_pipeline(
            latex_file=str(latex_path),
            code_paths=args.code,
            output_dir=args.output,
            graph_dir=args.graph
        ))

        print(f"\n✅ Pipeline completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())