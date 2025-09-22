import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from lightrag import QueryParam
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

from latex_processor import process_latex, extract_mathematical_concepts
from knowledge_graph_builder import KnowledgeGraphBuilder

load_dotenv()


class ResearchExtender:
    """Generate code and paper extensions using LLM assistance via OpenRouter"""

    def __init__(self, graph_builder: KnowledgeGraphBuilder, llm_model: str = "anthropic/claude-3.7-sonnet"):
        self.graph_builder = graph_builder
        self.rag = graph_builder.rag
        self.llm_model = llm_model
        
        # OpenRouter config
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not self.openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment")
            
        self.client = AsyncOpenAI(
            api_key=self.openrouter_key,
            base_url=self.openrouter_base,
        )

    async def find_gaps(self, latex_data: Dict, code_data: Dict, mappings: List[Dict]) -> List[Dict]:
        """Find gaps using RAG and LLM analysis"""

        # Use the existing gap detection from graph builder
        structural_gaps = self.graph_builder.find_implementation_gaps(latex_data, code_data, mappings)

        # Enhance with LLM analysis
        enhanced_gaps = []

        for gap in structural_gaps:
            if gap['type'] == 'missing_implementation':
                # Query RAG for related implementations
                query = f"How can equation {gap['equation']['label']} be implemented in code? What algorithms are needed?"
                context = await self.rag.aquery(query, param=QueryParam(mode="hybrid"))

                # Get LLM suggestions
                suggestion_prompt = f"""
                Equation: {gap['equation']['content']}
                Context from knowledge base: {context}

                Suggest a Python implementation approach for this equation. Include:
                1. Function signature
                2. Key algorithmic steps
                3. Required dependencies
                4. Computational complexity considerations
                """

                response = await self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": suggestion_prompt}],
                    max_tokens=1000,
                    temperature=0
                )

                gap['llm_suggestion'] = response.choices[0].message.content
                gap['related_context'] = context

            elif gap['type'] == 'missing_algorithm':
                # Query for algorithmic details
                query = f"What are the implementation details for {gap['section']['title']}? What code is needed?"
                context = await self.rag.aquery(query, param=QueryParam(mode="hybrid"))

                # Get LLM algorithm design
                algorithm_prompt = f"""
                Section: {gap['section']['title']}
                Content: {gap['section']['content'][:500]}...
                Context: {context}

                Design a complete algorithm implementation. Include:
                1. Class/function structure
                2. Core algorithm logic
                3. Input/output specifications
                4. Integration points with existing code
                """

                response = await self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": algorithm_prompt}],
                    max_tokens=1500,
                    temperature=0
                )

                gap['algorithm_design'] = response.choices[0].message.content
                gap['related_context'] = context

            enhanced_gaps.append(gap)

        return enhanced_gaps

    async def generate_code_extensions(self, gaps: List[Dict], output_dir: str = "./extensions") -> Dict[str, str]:
        """Generate code files for missing implementations"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        generated_files = {}

        for gap in gaps:
            if gap['type'] in ['missing_implementation', 'missing_algorithm']:

                # Generate detailed implementation
                if gap['type'] == 'missing_implementation':
                    impl_prompt = f"""
                    Generate complete Python code for this mathematical equation:
                    Equation: {gap['equation']['content']}
                    Label: {gap['equation']['label']}

                    Suggested approach: {gap.get('llm_suggestion', 'No suggestion available')}
                    Related context: {gap.get('related_context', 'No context available')}

                    Requirements:
                    1. Complete function implementation
                    2. Proper error handling
                    3. Type hints
                    4. Docstring with mathematical explanation
                    5. Unit tests
                    6. Example usage

                    Return only the Python code.
                    """
                    filename = f"impl_{gap['equation']['label'].replace(':', '_')}.py"

                else:  # missing_algorithm
                    impl_prompt = f"""
                    Generate complete Python code for this algorithm:
                    Title: {gap['section']['title']}
                    Description: {gap['section']['content'][:300]}...

                    Algorithm design: {gap.get('algorithm_design', 'No design available')}
                    Related context: {gap.get('related_context', 'No context available')}

                    Requirements:
                    1. Complete class/function implementation
                    2. Proper initialization and methods
                    3. Type hints and docstrings
                    4. Integration with existing codebase
                    5. Unit tests
                    6. Example usage

                    Return only the Python code.
                    """
                    section_name = gap['section']['title'].lower().replace(' ', '_').replace('-', '_')
                    filename = f"algorithm_{section_name}.py"

                # Generate code
                response = await self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": impl_prompt}],
                    max_tokens=3000,
                    temperature=0.1
                )

                code_content = response.choices[0].message.content

                # Clean up code content
                if "```python" in code_content:
                    code_content = code_content.split("```python")[1].split("```")[0]
                elif "```" in code_content:
                    code_content = code_content.split("```")[1].split("```")[0]

                # Write to file
                file_path = output_path / filename
                file_path.write_text(code_content, encoding='utf-8')
                generated_files[filename] = str(file_path)

        return generated_files

    async def generate_paper_extensions(self, gaps: List[Dict], latex_data: Dict, output_dir: str = "./extensions") -> Dict[str, str]:
        """Generate LaTeX extensions for new implementations"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        generated_sections = {}

        # Group gaps by type for coherent sections
        implementation_gaps = [g for g in gaps if g['type'] == 'missing_implementation']
        algorithm_gaps = [g for g in gaps if g['type'] == 'missing_algorithm']

        # Generate implementation details section
        if implementation_gaps:
            impl_prompt = f"""
            Generate a LaTeX section titled "Implementation Details" that describes the implementations for these equations:

            Paper context:
            Title: {latex_data.get('title', 'Research Paper')}
            Abstract: {latex_data.get('abstract', 'Not available')[:200]}...

            Missing implementations:
            {chr(10).join([f"- Equation {g['equation']['label']}: {g['equation']['content']}" for g in implementation_gaps])}

            Requirements:
            1. Formal mathematical description
            2. Algorithmic complexity analysis
            3. Implementation considerations
            4. Proper LaTeX formatting with equations, algorithms, references
            5. Academic writing style
            6. Citations to relevant work (use \\cite{} format)

            Return only the LaTeX section content.
            """

            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": impl_prompt}],
                max_tokens=2000,
                temperature=0.1
            )

            latex_content = response.choices[0].message.content
            file_path = output_path / "implementation_details.tex"
            file_path.write_text(latex_content, encoding='utf-8')
            generated_sections["implementation_details.tex"] = str(file_path)

        # Generate algorithms section
        if algorithm_gaps:
            algo_prompt = f"""
            Generate a LaTeX section titled "Extended Algorithms" for these missing algorithms:

            Paper context:
            Title: {latex_data.get('title', 'Research Paper')}

            Missing algorithms:
            {chr(10).join([f"- {g['section']['title']}: {g['section']['content'][:100]}..." for g in algorithm_gaps])}

            Requirements:
            1. Detailed algorithm descriptions using \\begin{algorithm} environments
            2. Pseudocode with proper formatting
            3. Complexity analysis
            4. Theoretical justification
            5. Comparison with existing methods
            6. Proper citations and references

            Return only the LaTeX section content.
            """

            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": algo_prompt}],
                max_tokens=2500,
                temperature=0.1
            )

            latex_content = response.choices[0].message.content
            file_path = output_path / "extended_algorithms.tex"
            file_path.write_text(latex_content, encoding='utf-8')
            generated_sections["extended_algorithms.tex"] = str(file_path)

        # Generate experimental section
        if gaps:
            exp_prompt = f"""
            Generate a LaTeX section titled "Experimental Validation" that describes experiments to validate the new implementations:

            Implementations added: {len(implementation_gaps)} equations, {len(algorithm_gaps)} algorithms

            Requirements:
            1. Experimental setup description
            2. Evaluation metrics and baselines
            3. Expected results and analysis
            4. Tables and figures placeholders with proper captions
            5. Statistical analysis approach
            6. Academic writing style

            Return only the LaTeX section content.
            """

            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": exp_prompt}],
                max_tokens=1500,
                temperature=0.1
            )

            latex_content = response.choices[0].message.content
            file_path = output_path / "experimental_validation.tex"
            file_path.write_text(latex_content, encoding='utf-8')
            generated_sections["experimental_validation.tex"] = str(file_path)

        return generated_sections

    async def generate_extensions(self, latex_file_path: str, code_paths: List[str], output_dir: str = "./extensions") -> Dict[str, Any]:
        """Complete extension generation pipeline"""

        # Process inputs
        latex_data = process_latex(latex_file_path)
        code_data = self.graph_builder.parse_code_files(code_paths)

        # Build knowledge graph
        await self.graph_builder.build_graph(latex_data, code_paths)

        # Create mappings and find gaps
        mappings = self.graph_builder.map_code_to_equations(code_data, latex_data)
        gaps = await self.find_gaps(latex_data, code_data, mappings)

        # Generate extensions
        code_files = await self.generate_code_extensions(gaps, output_dir)
        latex_files = await self.generate_paper_extensions(gaps, latex_data, output_dir)

        # Create summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "timestamp": timestamp,
            "input_paper": latex_file_path,
            "input_code_paths": code_paths,
            "gaps_found": len(gaps),
            "gaps_by_type": {
                "missing_implementation": len([g for g in gaps if g['type'] == 'missing_implementation']),
                "missing_algorithm": len([g for g in gaps if g['type'] == 'missing_algorithm']),
                "orphaned_code": len([g for g in gaps if g['type'] == 'orphaned_code'])
            },
            "generated_code_files": list(code_files.keys()),
            "generated_latex_files": list(latex_files.keys()),
            "total_extensions": len(code_files) + len(latex_files)
        }

        # Save summary
        summary_path = Path(output_dir) / f"extension_summary_{timestamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

        return {
            "summary": summary,
            "code_files": code_files,
            "latex_files": latex_files,
            "gaps": gaps
        }


async def main():
    """Example usage and CLI interface"""

    # Example file paths - update with actual files
    latex_file = "My_Thesis_3/My Thesis 2/Chapters/Chp1_Intro.tex"  # Use actual .tex file
    code_paths = ["thesis_code/", "src/"]  # Directories will be expanded to files
    output_dir = "./research_extensions"

    # Build knowledge graph
    print("Building knowledge graph...")
    builder = KnowledgeGraphBuilder(
        working_dir="./research_graph",
        llm_model="anthropic/claude-3.7-sonnet",  # OpenRouter model
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384
    )

    # Create extender
    extender = ResearchExtender(builder, llm_model="anthropic/claude-3.7-sonnet")

    # Generate extensions
    print("Generating extensions...")
    result = await extender.generate_extensions(latex_file, code_paths, output_dir)

    # Print summary
    print(f"\nExtension Summary:")
    print(f"Gaps found: {result['summary']['gaps_found']}")
    print(f"Code files generated: {len(result['code_files'])}")
    print(f"LaTeX files generated: {len(result['latex_files'])}")
    print(f"Output directory: {output_dir}")

    # List generated files
    print(f"\nGenerated files:")
    for filename in result['code_files']:
        print(f"  Code: {filename}")
    for filename in result['latex_files']:
        print(f"  LaTeX: {filename}")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
