"""
Simple test script for LaTeX Knowledge Graph Generator
"""
import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.latex_graph_generator import Config, GraphBuilder
from src.latex_graph_generator.main import generate_knowledge_graph

async def test_knowledge_graph():
    # Test with one of your thesis files
    latex_file = "My_Thesis_3/My Thesis 2/Chapters/Chp1_Intro.tex"

    if not Path(latex_file).exists():
        print(f"[ERROR] Test file not found: {latex_file}")
        return False

    print(f"[TESTING] with: {latex_file}")

    try:
        # Test the main function
        builder, result = await generate_knowledge_graph(
            latex_file,
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
    success = asyncio.run(test_knowledge_graph())
    if success:
        print("\n[PASSED] All tests passed!")
    else:
        print("\n[FAILED] Tests failed!")
        sys.exit(1)