#!/usr/bin/env python3
"""
Deep Research Discovery Crew - Phase 1 Implementation

This module implements the research discovery system that:
1. Queries dual KG system (code + LaTeX) for relevant concepts
2. Uses LinkUp deep search to find research papers
3. Presents 3-5 research options with summaries and relevance scores

Usage:
    from deep_research_crew import create_research_discovery_crew
    crew = create_research_discovery_crew()
    result = crew.kickoff(inputs={"query": "optimize C-core actuator using deep learning"})
"""

import os
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LIGHT_RAG_CODE_DIR = "./lightrag_code_index"
LIGHT_RAG_LATEX_DIR = "./lightrag_latex_index"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"

def get_llm_client():
    """Initialize and return the LLM client"""
    return LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434"
    )

class KGQueryInput(BaseModel):
    """Input schema for KG Query Tool."""
    query: str = Field(description="The research query to search in KG")
    kg_type: str = Field(description="Type of KG: 'code' or 'latex'")
    max_concepts: int = Field(default=20, description="Maximum concepts to extract")

class KGQueryTool(BaseTool):
    """Tool to query the dual KG system (code + LaTeX)"""
    name: str = "KG Query Tool"
    description: str = "Query the dual KG system to extract relevant concepts"
    args_schema: type = KGQueryInput

    def __init__(self):
        super().__init__()

    def _run(self, query: str, kg_type: str = "code", max_concepts: int = 20) -> str:
        """Query KG and extract relevant concepts"""
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import openai_complete_if_cache
            from lightrag.llm.ollama import ollama_embed
            from lightrag.utils import EmbeddingFunc

            # Determine KG directory
            kg_dir = LIGHT_RAG_CODE_DIR if kg_type == "code" else LIGHT_RAG_LATEX_DIR

            if not os.path.exists(kg_dir):
                return f"KG directory not found: {kg_dir}"

            # Initialize LightRAG
            async def llm_func(prompt, **kwargs):
                return await openai_complete_if_cache(
                    "openai/gpt-4o-mini",
                    prompt,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    **kwargs
                )

            rag = LightRAG(
                working_dir=kg_dir,
                llm_model_func=llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts,
                        embed_model=OLLAMA_EMBEDDING_MODEL,
                        host=OLLAMA_HOST,
                    ),
                ),
            )

            # Query KG with hybrid mode for comprehensive results
            query_param = QueryParam(mode="hybrid")
            response = rag.query(query, param=query_param)

            # Extract concepts from response
            concepts_prompt = f"""
            From the following KG response, extract the most relevant concepts for: {query}

            KG Response:
            {response}

            Extract maximum {max_concepts} key concepts that are most relevant to electromagnetic design,
            FEMM simulation, machine learning optimization, or related topics.

            Return as JSON list of concepts with relevance scores (0-10):
            [
                {{"concept": "electromagnetic actuator", "relevance": 9.5, "source": "code"}},
                {{"concept": "finite element method", "relevance": 8.2, "source": "latex"}}
            ]
            """

            concepts_response = rag.query(concepts_prompt, param=QueryParam(mode="global"))

            return concepts_response

        except Exception as e:
            return f"Error querying KG: {str(e)}"

class ResearchOption(BaseModel):
    """Schema for research option output"""
    id: int
    title: str
    authors: List[str]
    summary: str
    methodology: str
    femm_relevance: float
    complexity: str
    source_url: str
    key_contributions: List[str]

class ResearchDiscoveryInput(BaseModel):
    """Input schema for research discovery"""
    query: str = Field(description="User's research query")
    max_options: int = Field(default=5, description="Maximum research options to return")

def create_research_discovery_crew():
    """Create the research discovery crew with 3 specialized agents"""

    # Initialize tools
    kg_query_tool = KGQueryTool()
    linkup_search_tool = LinkUpSearchTool()

    # Get LLM client
    llm = get_llm_client()

    # Agent 1: KG Concept Extractor
    kg_analyst = Agent(
        role="Knowledge Graph Analyst",
        goal="Extract relevant concepts from dual KG system (code + LaTeX) for the user's research query",
        backstory="""You are an expert at analyzing knowledge graphs to extract the most relevant
        concepts for electromagnetic design and FEMM-related research. You understand both
        theoretical (LaTeX) and implementation (code) aspects of research.""",
        verbose=True,
        allow_delegation=False,
        tools=[kg_query_tool],
        llm=llm,
        max_iter=3,  # Guardrail: limit iterations
    )

    # Agent 2: Research Paper Searcher
    research_searcher = Agent(
        role="Academic Research Specialist",
        goal="Find the most relevant academic papers for electromagnetic/FEMM research using deep web search",
        backstory="""You are a specialist in academic research with deep knowledge of
        electromagnetic design, finite element methods, and machine learning applications.
        You can identify the most promising research directions for FEMM-based projects.""",
        verbose=True,
        allow_delegation=False,
        tools=[linkup_search_tool],
        llm=llm,
        max_iter=3,  # Guardrail: limit iterations
    )

    # Agent 3: Research Options Ranker
    research_ranker = Agent(
        role="Research Strategy Consultant",
        goal="Analyze and rank research options for FEMM implementation feasibility and impact",
        backstory="""You are an expert consultant who evaluates research papers for their
        practical implementation potential in FEMM environments. You understand the
        constraints and opportunities of finite element electromagnetic simulation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=2,  # Guardrail: limit iterations
    )

    # Task 1: Extract concepts from dual KG
    kg_extraction_task = Task(
        description="""
        Query both the code KG and LaTeX KG to extract relevant concepts for: {query}

        For each KG:
        1. Query with the user's research question
        2. Extract maximum 20 most relevant concepts
        3. Focus on electromagnetic design, FEMM, ML optimization, actuators, etc.
        4. Return concepts with relevance scores (0-10)

        Return as structured JSON with concepts from both KGs.
        """,
        expected_output="JSON with extracted concepts from both code and LaTeX KGs",
        agent=kg_analyst,
        context=[],
    )

    # Task 2: Search for research papers
    research_search_task = Task(
        description="""
        Using the extracted KG concepts, search for relevant academic research papers.

        Search Strategy:
        1. Use extracted concepts to formulate targeted search queries
        2. Focus on: electromagnetic design, FEMM, finite element optimization, ML applications
        3. Look for papers that mention FEMM, pyfemm, or finite element electromagnetic simulation
        4. Prioritize recent papers (2019+) with practical implementation potential
        5. Maximum 15 papers retrieved

        Return raw search results with paper metadata.
        """,
        expected_output="Raw search results with paper titles, authors, abstracts, URLs",
        agent=research_searcher,
        context=[kg_extraction_task],
    )

    # Task 3: Rank and summarize research options
    ranking_task = Task(
        description="""
        Analyze the search results and create 3-5 ranked research options.

        For each paper, evaluate:
        - FEMM relevance (0-10): How directly applicable to FEMM implementation?
        - Implementation complexity: Low/Medium/High
        - Key methodology: What approach does the paper propose?
        - Practical value: How useful for real FEMM projects?

        Return exactly 5 options ranked by FEMM relevance score.
        Each option must include:
        - Title, authors, summary, methodology
        - FEMM relevance score (0-10)
        - Complexity level (Low/Medium/High)
        - Source URL
        - Key contributions for FEMM implementation

        Format as structured JSON.
        """,
        expected_output="JSON with 5 ranked research options",
        agent=research_ranker,
        context=[research_search_task],
    )

    # Create the crew
    crew = Crew(
        agents=[kg_analyst, research_searcher, research_ranker],
        tasks=[kg_extraction_task, research_search_task, ranking_task],
        verbose=True,
        process=Process.sequential,
        planning=False,  # Disable planning for faster execution
    )

    return crew

# Import LinkUpSearchTool from agents.py
class LinkUpSearchInput(BaseModel):
    """Input schema for LinkUp Search Tool."""
    query: str = Field(description="The search query to perform")
    depth: str = Field(default="deep", description="Depth of search: 'standard' or 'deep'")
    output_type: str = Field(default="searchResults", description="Output type")

class LinkUpSearchTool(BaseTool):
    name: str = "LinkUp Search"
    description: str = "Search the web for information using LinkUp and return comprehensive results"
    args_schema: type = LinkUpSearchInput

    def __init__(self):
        super().__init__()

    def _run(self, query: str, depth: str = "deep", output_type: str = "searchResults") -> str:
        """Execute LinkUp search and return results."""
        try:
            from linkup import LinkupClient

            # Initialize LinkUp client
            linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))

            # Perform deep search for academic papers
            search_response = linkup_client.search(
                query=query,
                depth=depth,
                output_type=output_type
            )

            return str(search_response)
        except Exception as e:
            return f"Error occurred while searching: {str(e)}"

def run_research_discovery(query: str, max_options: int = 5) -> str:
    """Main function to run research discovery"""
    try:
        crew = create_research_discovery_crew()
        result = crew.kickoff(inputs={
            "query": query,
            "max_options": max_options
        })

        # Parse and validate the result
        try:
            # Extract JSON from the result
            result_text = result.raw
            # For now, return the raw result - in production we'd parse and validate
            return result_text
        except:
            return result.raw

    except Exception as e:
        return f"Error in research discovery: {str(e)}"

if __name__ == "__main__":
    # Test the research discovery
    test_query = "optimize C-core actuator using deep learning"
    print(f"Testing research discovery with query: {test_query}")
    result = run_research_discovery(test_query)
    print("Result:", result)
