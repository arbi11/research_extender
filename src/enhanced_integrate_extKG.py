#!/usr/bin/env python3
"""
Enhanced KG Integration with LightRAG - OpenRouter LLM + Ollama Embeddings

This script:
1. Loads the KG JSON file with entities, relationships, and chunks (EXACTLY like sample_code)
2. Initializes LightRAG with OpenRouter LLM + Ollama embeddings
3. Inserts the custom KG into LightRAG (EXACTLY like sample_code)
4. ONLY CHANGES: OpenRouter LLM + meaningful chunk IDs

Usage:
    python enhanced_integrate_extKG.py --kg-file kg_pALM_1001_1932.json
"""

import os
import json
import argparse
import asyncio
from pathlib import Path
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Configuration - OpenRouter for LLM, Ollama for embeddings only
OLLAMA_HOST = "http://localhost:11434"  # For embeddings only
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
OLLAMA_EMBEDDING_DIM = 1024

OPENROUTER_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-5")

def load_kg(kg_file: str) -> dict:
    """Load knowledge graph from JSON file - EXACTLY like sample_code"""
    print(f"Loading KG from: {kg_file}")
    with open(kg_file, 'r', encoding='utf-8') as f:
        kg = json.load(f)

    print(f"  Entities: {len(kg.get('entities', []))}")
    print(f"  Relationships: {len(kg.get('relationships', []))}")
    print(f"  Chunks: {len(kg.get('chunks', []))}")

    return kg

async def initialize_lightrag(working_dir: str):
    """Initialize LightRAG with OpenRouter LLM + Ollama embeddings"""
    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)

    print(f"\nInitializing LightRAG...")
    print(f"  Working directory: {working_dir}")
    print(f"  LLM: OpenRouter + {OPENROUTER_LLM_MODEL}")
    print(f"  Embeddings: Ollama + {OLLAMA_EMBEDDING_MODEL}")

    # OpenRouter LLM function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable required")

        return await openai_complete_if_cache(
            OPENROUTER_LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            **kwargs,
        )

    # Create LightRAG instance with OpenRouter LLM + Ollama embeddings
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=OPENROUTER_LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=OLLAMA_EMBEDDING_DIM,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=OLLAMA_EMBEDDING_MODEL,
                host=OLLAMA_HOST,
            ),
        ),
        default_embedding_timeout=180, # Set timeout to 3 minutes
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def main(kg_file: str, output_dir: str):
    """Main function - Enhanced with OpenRouter LLM"""
    print("=" * 60)
    print("ENHANCED KG INTEGRATION WITH LIGHTRAG")
    print("=" * 60)

    # Load KG - EXACTLY like sample_code
    kg = load_kg(kg_file)

    # Check if chunks exist - EXACTLY like sample_code
    if 'chunks' not in kg or len(kg['chunks']) == 0:
        print("\n⚠️  No chunks found in KG!")
        print("Run enhanced_chunks.py first:")
        print(f"  python enhanced_chunks.py --kg-file {kg_file}")
        return

    # Initialize LightRAG with OpenRouter LLM
    rag = await initialize_lightrag(output_dir)

    # Insert custom KG - EXACTLY like sample_code
    print("\n" + "=" * 60)
    print("INSERTING CUSTOM KG INTO LIGHTRAG")
    print("=" * 60)

    await rag.ainsert_custom_kg(kg)

    print("\n✅ Custom KG successfully integrated with LightRAG!")
    print(f"   Working directory: {output_dir}")
    print(f"   LLM: OpenRouter + {OPENROUTER_LLM_MODEL}")
    print(f"   Embeddings: Ollama + {OLLAMA_EMBEDDING_MODEL}")
    print(f"   Ollama host: {OLLAMA_HOST}")
    print("\nYou can now query the KG using enhanced_interface.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate custom KG with LightRAG - OpenRouter LLM + Ollama embeddings")
    parser.add_argument("--kg-file", required=True, help="Path to KG JSON file (e.g., KG_code/KG_code_ALL/graph.json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for LightRAG index (e.g., ./lightrag_code_index)")

    args = parser.parse_args()

    asyncio.run(main(args.kg_file, args.output_dir))
