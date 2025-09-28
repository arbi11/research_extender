#!/usr/bin/env python3
"""
Clean CLI Knowledge Assistant

Simple, transparent tool for querying codebase knowledge with GPT-5.
No unnecessary complexity or error handling.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import openai
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-5")

class CleanKnowledgeAssistant:
    def __init__(self, kg_directory: str):
        self.kg_directory = Path(kg_directory)
        self.entities = {}
        self.relationships = {}
        self.client = openai.OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )
        self.model = DEFAULT_MODEL

        # Initialize LightRAG with OpenRouter-backed LLM and Ollama embeddings
        async def _llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
            return await openai_complete_if_cache(
                self.model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                **kwargs,
            )

        # Configure Ollama embedding backend (e.g., embeddinggemma)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        embed_model = os.getenv("EMBEDDING_MODEL", "embeddinggemma")

        def _embed_func(texts):
            # Return coroutine; LightRAG handles awaiting in async contexts
            return ollama_embed(texts, embed_model=embed_model, host=ollama_host)

        # Set embedding dimension from env (match embeddinggemma default=768)
        embed_dim = int(os.getenv("EMBEDDING_DIM", "768"))

        workdir = str(self.kg_directory / ".lightrag_work")
        # Clean stale LightRAG workspace to avoid GraphML parse errors and embedding dim mismatch
        if os.path.exists(workdir):
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)

        self.rag = LightRAG(
            working_dir=workdir,
            llm_model_func=_llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embed_dim,
                max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
                func=_embed_func
            )
        )
        self._kg_loaded = False

    async def _initialize_kg_async(self):
        """Initialize KG in single async context"""
        if self._kg_loaded:
            return
            
        # Load JSON files
        entities_file = self.kg_directory / "entities.json"
        relationships_file = self.kg_directory / "relationships.json"
        graph_file = self.kg_directory / "graph.json"
        
        with open(entities_file) as f:
            self.entities = {e["entity_name"]: e for e in json.load(f)}
        with open(relationships_file) as f:
            self.relationships = {(r["src_id"], r["tgt_id"]): r for r in json.load(f)}
        
        if graph_file.exists():
            with open(graph_file, "r", encoding="utf-8") as gf:
                kg = json.load(gf)
            if "chunks" not in kg:
                kg["chunks"] = []
            
            # All async operations in single context
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            await self.rag.ainsert_custom_kg(kg)
            self._kg_loaded = True

    def load_knowledge_graph(self):
        """Load entities and relationships"""
        # Use single async context for all operations
        asyncio.run(self._initialize_kg_async())

    def find_relevant_entities(self, query: str) -> List[Dict]:
        """Find entities relevant to the query"""
        relevant = []
        query_lower = query.lower()

        for entity in self.entities.values():
            if (query_lower in entity["entity_name"].lower() or
                query_lower in entity["description"].lower()):
                relevant.append(entity)

        return relevant[:5]

    def find_relevant_relationships(self, entities_used: List[str]) -> List[Dict]:
        """Find relationships involving the used entities"""
        relevant = []

        for (src_id, tgt_id), relationship in self.relationships.items():
            if src_id in entities_used or tgt_id in entities_used:
                relevant.append(relationship)

        return relevant[:10]

    async def _query_async(self, query: str) -> str:
        """Query LightRAG in async context"""
        # Ensure KG is loaded first
        await self._initialize_kg_async()

        # Query with error handling
        try:
            resp = await self.rag.aquery(
                query,
                param=QueryParam(mode="global")
            )
            return resp if resp else "No response generated from knowledge graph."
        except Exception as e:
            print(f"LightRAG error: {e}")
            # Fallback to original method if LightRAG fails
            context = self.build_context(self.find_relevant_entities(query), self.find_relevant_relationships([e['entity_name'] for e in self.find_relevant_entities(query)]))
            fallback_prompt = f"""
            Answer this question about a magnetic field prediction codebase:

            Question: {query}

            Relevant context:
            {context}

            Provide a clear, accurate answer based on the context above.
            """
            fallback_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": fallback_prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            return fallback_resp.choices[0].message.content

    def _query_fallback(self, query: str) -> str:
        """Fallback query method using original approach"""
        # Find relevant entities and relationships
        relevant_entities = self.find_relevant_entities(query)
        relevant_relationships = self.find_relevant_relationships([e['entity_name'] for e in relevant_entities])

        # Build context
        context = self.build_context(relevant_entities, relevant_relationships)

        # Generate response using OpenRouter directly
        prompt = f"""
        Answer this question about a magnetic field prediction codebase:

        Question: {query}

        Relevant context:
        {context}

        Provide a clear, accurate answer based on the context above.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content

    def generate_response(self, query: str, relevant_entities: List, relevant_relationships: List) -> str:
        """Generate response using GPT-5 with LightRAG fallback"""
        # Try LightRAG first, fallback to original method if it fails
        try:
            resp = asyncio.run(self._query_async(query))
            # Check if LightRAG returned a valid response
            if resp and resp != "No response generated from knowledge graph.":
                return resp
        except Exception as e:
            print(f"LightRAG failed, using fallback: {e}")

        # Fallback to original method
        return self._query_fallback(query)

    def build_context(self, entities: List, relationships: List) -> str:
        """Build context string"""
        context_parts = []

        for entity in entities:
            context_parts.append(f"""
{entity['entity_name']} ({entity['entity_type']}):
{entity['description']}
""")

        for rel in relationships:
            context_parts.append(f"""
{rel['src_id']} → {rel['tgt_id']}: {rel['description']}
""")

        return "\n".join(context_parts)

    def display_response(self, query: str, response: str, entities_used: List, relationships_used: List):
        """Display response with source transparency"""
        print(f"\n🤖 Response:")
        print(response)

        print("\n📊 Sources Used:")
        print("Entities:")
        for entity in entities_used:
            print(f"  • {entity['entity_name']}")

        print("Relationships:")
        for rel in relationships_used:
            print(f"  • {rel['src_id']} → {rel['tgt_id']}")

    def run(self):
        """Main CLI loop"""
        print("🔍 Codebase Assistant")
        print("Ask questions about your magnetic field prediction code:")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit']:
                break

            # Load knowledge graph
            self.load_knowledge_graph()

            # Find relevant information
            relevant_entities = self.find_relevant_entities(query)
            relevant_relationships = self.find_relevant_relationships([e['entity_name'] for e in relevant_entities])

            # Generate and display response
            response = self.generate_response(query, relevant_entities, relevant_relationships)
            self.display_response(query, response, relevant_entities, relevant_relationships)

if __name__ == "__main__":
    import sys
    kg_dir = sys.argv[1] if len(sys.argv) > 1 else "KG_code/KG_Chp2_FieldDistribution/"
    assistant = CleanKnowledgeAssistant(kg_dir)
    assistant.run()
