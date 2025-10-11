"""
Process the complete thesis PDF into a single comprehensive knowledge graph using RAG-Anything + LightRAG
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc

from .graph_builder import GraphBuilder

load_dotenv()

async def build_comprehensive_knowledge_graph(pdf_filename: str):
    # Working directory override for process_all.py
    working_dir = "./comprehensive_thesis_kg"

    # Model functions - validate required environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is required")
        print("Please set it in your .env file")
        return None, None

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def llm_func(prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    # Embedding configuration from environment
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
    embedding_dim = int(os.getenv("OLLAMA_EMBEDDING_DIM", "768"))
    embedding_max_tokens = int(os.getenv("OLLAMA_EMBEDDING_MAX_TOKENS", "8192"))

    print(f"Using embedding model: {embedding_model} (dim: {embedding_dim})")

    async def embedding_func(texts):
        import httpx

        embeddings = []
        for text in texts:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ollama_host}/api/embeddings",
                    json={
                        "model": embedding_model,
                        "prompt": text
                    }
                )
                embedding = response.json()["embedding"]
                embeddings.append(embedding)

        return embeddings

    embedding_wrapper = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=embedding_max_tokens,
        func=embedding_func
    )

    # Use the specified PDF from thesis_pdf directory
    pdf_file = Path(__file__).parents[2].absolute() / "thesis_pdf" / pdf_filename

    if not pdf_file.exists():
        print(f"ERROR: PDF file not found: {pdf_file}")
        print(f"Please ensure '{pdf_filename}' exists in the 'thesis_pdf/' directory")
        return None, None

    print(f"Processing complete thesis PDF: {pdf_file.name}")
    print(f"File size: {pdf_file.stat().st_size / (1024*1024):.1f} MB")

    # Build single graph builder
    builder = GraphBuilder(working_dir)

    print(f"\nBuilding comprehensive knowledge graph from PDF...")

    result = await builder.build(str(pdf_file), llm_func, embedding_wrapper)

    print(f"   Sections: {result['total_sections']}")
    print(f"   Equations: {result['total_equations']}")
    print(f"   Relationships: {result['total_relationships']}")
    print(f"   Relationship Types: {result['relationship_types']}")

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE KNOWLEDGE GRAPH BUILT SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Files processed: 1/1")
    print(f"Total Statistics:")
    print(f"   Total Sections: {result['total_sections']}")
    print(f"   Total Equations: {result['total_equations']}")
    print(f"   Total Relationships: {result['total_relationships']}")
    print(f"   Relationship Types: {', '.join(result['relationship_types'])}")
    print(f"Graph stored in: {working_dir}")

    return builder, {
        'processed_files': [pdf_filename],
        'total_sections': result['total_sections'],
        'total_equations': result['total_equations'],
        'total_relationships': result['total_relationships'],
        'relationship_types': result['relationship_types'],
        'graph_directory': builder.working_dir
    }

async def test_queries(builder):
    """Test some interesting queries on the comprehensive knowledge graph"""

    test_queries = [
        "What are the main mathematical concepts discussed across all chapters?",
        "Find all equations related to neural networks",
        "What theorems and proofs are presented in the thesis?",
        "How are optimization algorithms used in different chapters?",
        "What are the key performance metrics and evaluation methods?"
    ]

    print(f"\nTesting queries on comprehensive knowledge graph:")
    print("="*50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        answer = await builder.query(query)
        print(f"Answer: {answer[:300]}...")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build comprehensive knowledge graph from thesis PDF")
    parser.add_argument("pdf_filename", help="Name of PDF file in thesis_pdf/ directory (e.g., 'my_thesis.pdf')")
    parser.add_argument("--no-test", action="store_true", help="Skip test queries after building")

    args = parser.parse_args()

    async def main():
        print(f"Building knowledge graph from: {args.pdf_filename}")
        builder, result = await build_comprehensive_knowledge_graph(args.pdf_filename)

        if builder is None:
            return 1

        # Test some queries unless skipped
        if not args.no_test:
            await test_queries(builder)

        return 0

    exit_code = asyncio.run(main())
    exit(exit_code)