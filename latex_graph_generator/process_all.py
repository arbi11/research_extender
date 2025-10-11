"""
Process the complete thesis PDF into a single comprehensive knowledge graph using RAG-Anything + LightRAG
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc

from src.latex_graph_generator.config import Config
from src.latex_graph_generator.graph_builder import GraphBuilder

load_dotenv()

async def build_comprehensive_knowledge_graph():
    config = Config(working_dir="./comprehensive_thesis_kg")

    # Model functions
    api_key = os.getenv("OPENROUTER_API_KEY")
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

    async def embedding_func(texts):
        import httpx

        embeddings = []
        for text in texts:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": "nomic-embed-text:latest",
                        "prompt": text
                    }
                )
                embedding = response.json()["embedding"]
                embeddings.append(embedding)

        return embeddings

    embedding_wrapper = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=embedding_func
    )

    # Use the complete thesis PDF
    pdf_file = Path("My_Thesis_3/Khan_Arbaaz_ECE_Thesis.pdf")

    if not pdf_file.exists():
        print(f"ERROR: PDF file not found: {pdf_file}")
        return None, None

    print(f"Processing complete thesis PDF: {pdf_file.name}")
    print(f"File size: {pdf_file.stat().st_size / (1024*1024):.1f} MB")

    # Build single graph builder
    builder = GraphBuilder(config)

    print(f"\nBuilding comprehensive knowledge graph from PDF...")

    # Initialize LightRAG
    builder.lightrag = builder.processor.lightrag = None

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
    print(f"Graph stored in: {config.working_dir}")

    return builder, {
        'processed_files': ['Khan_Arbaaz_ECE_Thesis.pdf'],
        'total_sections': result['total_sections'],
        'total_equations': result['total_equations'],
        'total_relationships': result['total_relationships'],
        'relationship_types': result['relationship_types'],
        'graph_directory': config.working_dir
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
    async def main():
        builder, result = await build_comprehensive_knowledge_graph()

        # Test some queries
        await test_queries(builder)

        return 0

    exit_code = asyncio.run(main())
    exit(exit_code)