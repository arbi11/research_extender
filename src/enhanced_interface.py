#!/usr/bin/env python3
"""
Enhanced LightRAG Query Interface - 3-Response System

Interactive CLI with dual-KG support:
1. Model Selection - Claude, Gemini, GPT-5, or Custom
2. Query Loop - Gets 3 responses for each query:
   - Response 1: Code KG (implementation details)
   - Response 2: LaTeX KG (theory/equations)
   - Response 3: Combined synthesis

Usage:
    python src/enhanced_interface.py
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
OLLAMA_EMBEDDING_DIM = 1024

# Working directories
CODE_KG_DIR = "./lightrag_code_index"
LATEX_KG_DIR = "./lightrag_latex_index"

def select_llm_model():
    """Let user select LLM model"""
    print("\n" + "=" * 60)
    print("SELECT LLM MODEL:")
    print("=" * 60)
    print("1. Claude       - anthropic/claude-sonnet-4.5")
    print("2. Gemini       - google/gemini-2.5-flash")
    print("3. GPT-5        - openai/gpt-5")
    print("4. Custom       - Enter your own OpenRouter model ID")
    print()

    while True:
        choice = input("Enter your choice (1-4, default=1): ").strip()

        if choice == "" or choice == "1":
            return "anthropic/claude-sonnet-4.5"
        elif choice == "2":
            return "google/gemini-2.5-flash"
        elif choice == "3":
            return "openai/gpt-5"
        elif choice == "4":
            custom_model = input("Enter OpenRouter model ID (e.g., 'x-ai/glm-4.6'): ").strip()
            return custom_model if custom_model else "anthropic/claude-sonnet-4.5"
        else:
            print("❌ Invalid choice. Please enter 1-4.")

async def llm_model_func(selected_model, prompt, system_prompt=None, history_messages=[], **kwargs):
    """Dynamic OpenRouter LLM function"""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment variable required")

    return await openai_complete_if_cache(
        selected_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        **kwargs,
    )

async def initialize_single_rag(working_dir: str, kg_name: str, selected_model: str):
    """Initialize a single LightRAG instance"""
    if not os.path.exists(working_dir):
        print(f"❌ Error: {kg_name} KG index not found at {working_dir}")
        print(f"   Please run integration first!")
        return None

    print(f"🔧 Initializing {kg_name} KG...")
    print(f"   Directory: {working_dir}")

    async def dynamic_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await llm_model_func(selected_model, prompt, system_prompt, history_messages, **kwargs)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=dynamic_llm_func,
        llm_model_name=selected_model,
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

def select_mode():
    """Let user select query mode"""
    print("\n" + "=" * 60)
    print("SELECT QUERY MODE:")
    print("=" * 60)
    print("1. Naive   - Vector search on chunks")
    print("2. Local   - Local graph traversal")
    print("3. Global  - Global graph analysis")
    print("4. Hybrid  - Combination (RECOMMENDED)")
    print()

    while True:
        choice = input("Enter your choice (1-4, default=4): ").strip()

        if choice == "" or choice == "4":
            return "hybrid"
        elif choice == "1":
            return "naive"
        elif choice == "2":
            return "local"
        elif choice == "3":
            return "global"
        else:
            print("❌ Invalid choice. Please enter 1-4.")

async def query_with_synthesis(code_rag, latex_rag, query: str, mode: str, selected_model: str):
    """Query both KGs and synthesize combined response"""
    print(f"\n⏳ Querying knowledge graphs (mode: {mode})...")
    
    query_param = QueryParam(mode=mode, only_need_context=False)
    
    # Query 1: Code KG
    print("   🔍 Querying Code KG...")
    code_response = await code_rag.aquery(query, param=query_param) if code_rag else "Code KG not available"
    
    # Query 2: LaTeX KG
    print("   🔍 Querying LaTeX KG...")
    latex_response = await latex_rag.aquery(query, param=query_param) if latex_rag else "LaTeX KG not available"
    
    # Query 3: Synthesize combined
    print("   🔄 Synthesizing combined response...")
    synthesis_prompt = f"""Given a user question about a research thesis, I have gathered two perspectives:

CODE PERSPECTIVE (Implementation):
{code_response}

LATEX PERSPECTIVE (Theory/Paper):
{latex_response}

USER QUESTION: {query}

Please provide a comprehensive answer that synthesizes both perspectives, explaining:
1. The theoretical foundation (from LaTeX)
2. The implementation details (from Code)
3. How theory and implementation connect

Be concise but thorough."""

    combined_response = await llm_model_func(
        selected_model,
        synthesis_prompt,
        system_prompt="You are a helpful research assistant synthesizing information from code and academic papers."
    )
    
    return code_response, latex_response, combined_response

def display_responses(query: str, code_resp: str, latex_resp: str, combined_resp: str, mode: str):
    """Display all 3 responses formatted"""
    print("\n" + "=" * 60)
    print("QUERY RESULTS")
    print("=" * 60)
    print(f"Question: {query}")
    print(f"Mode: {mode.upper()}")
    
    print("\n" + "-" * 60)
    print("📝 RESPONSE 1: CODE KG (Implementation)")
    print("-" * 60)
    print(code_resp)
    
    print("\n" + "-" * 60)
    print("📚 RESPONSE 2: LATEX KG (Theory/Paper)")
    print("-" * 60)
    print(latex_resp)
    
    print("\n" + "-" * 60)
    print("🔗 RESPONSE 3: COMBINED SYNTHESIS")
    print("-" * 60)
    print(combined_resp)
    
    print("\n" + "=" * 60)

async def main():
    """Main CLI loop with 3-response system"""
    print("=" * 60)
    print("ENHANCED LIGHTRAG - 3-RESPONSE QUERY SYSTEM")
    print("=" * 60)

    # Step 1: Select model
    selected_model = select_llm_model()
    print(f"\n✅ Selected model: {selected_model}")

    # Step 2: Initialize both KGs
    print("\n" + "=" * 60)
    print("INITIALIZING KNOWLEDGE GRAPHS")
    print("=" * 60)
    
    code_rag = await initialize_single_rag(CODE_KG_DIR, "CODE", selected_model)
    latex_rag = await initialize_single_rag(LATEX_KG_DIR, "LATEX", selected_model)
    
    if code_rag is None and latex_rag is None:
        print("\n❌ No KGs available! Please run integration first.")
        return
    
    if code_rag:
        print(f"✅ Code KG initialized: {CODE_KG_DIR}")
    if latex_rag:
        print(f"✅ LaTeX KG initialized: {LATEX_KG_DIR}")

    # Step 3: Select query mode
    mode = select_mode()
    print(f"\n✅ Selected mode: {mode.upper()}")
    print("\nTip: Type 'mode' to change query mode")
    print("     Type 'quit' or 'exit' to exit")

    # Query loop
    while True:
        print("\n" + "-" * 60)
        query = input("\n🔍 Enter your question: ").strip()

        if not query:
            continue

        # Check for special commands
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if query.lower() == 'mode':
            mode = select_mode()
            print(f"\n✅ Mode changed to: {mode.upper()}")
            continue

        # Execute 3-response query
        code_resp, latex_resp, combined_resp = await query_with_synthesis(
            code_rag, latex_rag, query, mode, selected_model
        )
        
        # Display results
        display_responses(query, code_resp, latex_resp, combined_resp, mode)

if __name__ == "__main__":
    asyncio.run(main())
