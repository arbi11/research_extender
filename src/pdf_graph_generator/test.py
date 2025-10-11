"""
Simple test script for PDF Knowledge Graph Generator
"""
import asyncio
import sys
from pathlib import Path

# Add the project root to Python path (allows imports from anywhere)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pdf_graph_generator import GraphBuilder
from src.pdf_graph_generator.main import generate_knowledge_graph

async def test_knowledge_graph(pdf_filename: str = None):
    # Test with a thesis PDF file from thesis_pdf directory
    if pdf_filename is None:
        # Look for any PDF in thesis_pdf directory
        thesis_pdf_dir = Path("thesis_pdf")
        if not thesis_pdf_dir.exists():
            print(f"[ERROR] thesis_pdf directory not found")
            return False

        pdf_files = list(thesis_pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[ERROR] No PDF files found in thesis_pdf/ directory")
            return False

        pdf_file = pdf_files[0]
        print(f"[AUTO-SELECTED] Found PDF: {pdf_file.name}")
    else:
        pdf_file = Path("thesis_pdf") / pdf_filename

    if not pdf_file.exists():
        print(f"[ERROR] Test file not found: {pdf_file}")
        return False

    print(f"[TESTING] with: {pdf_file}")

    try:
        # Test the main function
        builder, result = await generate_knowledge_graph(
            str(pdf_file),
            working_dir="./test_kg_output"
        )

        print(f"[SUCCESS] Test completed successfully!")
        print(f"[RESULTS]:")
        print(f"   Sections: {result['total_sections']}")
        print(f"   Equations: {result['total_equations']}")
        print(f"   Relationships: {result['total_relationships']}")
        print(f"   Relationship types: {result['relationship_types']}")

        # Test a simple query
        query_result = await builder.query("What are the main topics discussed?")
        print(f"\n[QUERY] Sample Query Result:")
        print(f"{query_result[:200]}...")

        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PDF Knowledge Graph Generator")
    parser.add_argument("--pdf", help="Name of PDF file in thesis_pdf/ directory (optional, will auto-detect if not provided)")

    args = parser.parse_args()

    success = asyncio.run(test_knowledge_graph(args.pdf))
    if success:
        print("\n[PASSED] All tests passed!")
    else:
        print("\n[FAILED] Tests failed!")
        sys.exit(1)