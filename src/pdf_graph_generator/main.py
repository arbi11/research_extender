"""
python -m src.pdf_graph_generator.main "Khan_Arbaaz_ECE_Thesis.pdf" --output "lightrag_pdf_index"
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc

from .graph_builder import GraphBuilder

load_dotenv()

async def generate_knowledge_graph(pdf_file: str, working_dir: str = None):
    # Use provided working_dir or environment variable or default
    final_working_dir = working_dir or os.getenv("PDF_KG_WORKING_DIR", "./pdf_kg_output")

    # Model functions
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.5-flash-lite")

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

    async def embedding_func(texts):
        # Use Ollama for embeddings
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

    # Build graph
    builder = GraphBuilder(final_working_dir)
    result = await builder.build(pdf_file, llm_func, embedding_wrapper)

    return builder, result

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF Knowledge Graph Generator")
    parser.add_argument("pdf_filename", help="Name of PDF file in thesis_pdf/ directory")
    parser.add_argument("--output", "-o", default="lightrag_pdf_index", help="Output directory name in root")
    parser.add_argument("--query", "-q", help="Query the graph after building")

    args = parser.parse_args()

    async def main():
        # Construct paths relative to project root
        pdf_file = Path(__file__).parents[2].absolute() / "thesis_pdf" / args.pdf_filename
        output_dir = Path(__file__).parents[2].absolute() / args.output

        if not pdf_file.exists():
            print(f"Error: PDF file not found: {pdf_file}")
            print(f"Please ensure '{args.pdf_filename}' exists in the 'thesis_pdf/' directory")
            return 1

        print(f"Building knowledge graph from {args.pdf_filename}...")
        builder, result = await generate_knowledge_graph(str(pdf_file), str(output_dir))

        print(f"\n✅ Knowledge graph built successfully!")
        print(f"📊 Statistics:")
        print(f"   Sections: {result['total_sections']}")
        print(f"   Equations: {result['total_equations']}")
        print(f"   Mathematical relationships: {result['total_relationships']}")
        print(f"   Relationship types: {', '.join(result['relationship_types'])}")
        print(f"📁 Graph stored in: {result['graph_directory']}")

        if args.query:
            print(f"\n🔍 Query: {args.query}")
            answer = await builder.query(args.query)
            print(f"Answer: {answer}")

        return 0

    exit_code = asyncio.run(main())
    exit(exit_code)
