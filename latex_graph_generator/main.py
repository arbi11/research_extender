import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from lightrag.utils import EmbeddingFunc

from .config import Config
from .graph_builder import GraphBuilder

load_dotenv()

async def generate_knowledge_graph(latex_file: str, working_dir: str = None):
    config = Config(working_dir=working_dir or "./latex_kg_output")

    # Model functions
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("LLM_MODEL", "z-ai/glm-4.5-air:free")

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
        # Use Ollama for embeddings
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

    # Build graph
    builder = GraphBuilder(config)
    result = await builder.build(latex_file, llm_func, embedding_wrapper)

    return builder, result

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaTeX Knowledge Graph Generator")
    parser.add_argument("latex_file", help="Path to LaTeX file")
    parser.add_argument("--output", "-o", default="./latex_kg_output", help="Output directory")
    parser.add_argument("--query", "-q", help="Query the graph after building")

    args = parser.parse_args()

    async def main():
        latex_path = Path(args.latex_file)
        if not latex_path.exists():
            print(f"Error: LaTeX file not found: {args.latex_file}")
            return 1

        print(f"Building knowledge graph from {args.latex_file}...")
        builder, result = await generate_knowledge_graph(str(latex_path), args.output)

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