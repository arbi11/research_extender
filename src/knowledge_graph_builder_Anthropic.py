import ast
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx
import re

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.hf import hf_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

from transformers import AutoModel, AutoTokenizer
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()


class KnowledgeGraphBuilder:
    """Build unified knowledge graph from LaTeX and code"""

    def __init__(self, working_dir="./research_graph", llm_model="claude-3-7-sonnet-20250219"):
        self.working_dir = Path(working_dir)
        self.llm_model = llm_model
        self.embedding_model = "infly/inf-retriever-v1-1.5b"
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.embed_model = AutoModel.from_pretrained(self.embedding_model)

        # Create working directory
        os.makedirs(self.working_dir, exist_ok=True)

        # Initialize LightRAG
        self.rag = None

    async def llm_model_func(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        """LLM function for LightRAG"""
        client = AsyncAnthropic(api_key=self.anthropic_key)

        messages = []
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        request_params = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4000)
        }

        if system_prompt:
            request_params["system"] = system_prompt

        response = await client.messages.create(**request_params)
        return response.content[0].text

    async def initialize_rag(self):
        """Initialize the LightRAG system"""
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.llm_model_func,
            llm_model_name=self.llm_model,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=5000,
                func=lambda texts: hf_embed(
                    texts,
                    tokenizer=self.tokenizer,
                    embed_model=self.embed_model,
                ),
            ),
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    def parse_code_files(self, code_paths: List[str]) -> Dict[str, Any]:
        """Parse code files and extract structure"""

        code_data = {
            'files': [],
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'algorithms': []
        }

        for code_path in code_paths:
            file_path = Path(code_path)

            # Read file content
            content = file_path.read_text(encoding='utf-8')

            file_info = {
                'path': str(file_path),
                'name': file_path.name,
                'content': content,
                'size': len(content)
            }

            # Parse Python files
            if file_path.suffix == '.py':
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = {
                            'name': node.name,
                            'file': str(file_path),
                            'docstring': ast.get_docstring(node) or '',
                            'args': [arg.arg for arg in node.args.args],
                            'line_number': node.lineno,
                            'content': ast.get_source_segment(content, node) or ''
                        }
                        code_data['functions'].append(func_info)

                    elif isinstance(node, ast.ClassDef):
                        class_info = {
                            'name': node.name,
                            'file': str(file_path),
                            'docstring': ast.get_docstring(node) or '',
                            'line_number': node.lineno,
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                            'content': ast.get_source_segment(content, node) or ''
                        }
                        code_data['classes'].append(class_info)

                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            code_data['imports'].append({
                                'module': alias.name,
                                'alias': alias.asname,
                                'file': str(file_path)
                            })

                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            code_data['imports'].append({
                                'module': f"{module}.{alias.name}",
                                'alias': alias.asname,
                                'file': str(file_path)
                            })

            code_data['files'].append(file_info)

        return code_data

    def map_code_to_equations(self, code_data: Dict, latex_data: Dict) -> List[Dict]:
        """Map code implementations to mathematical equations"""

        mappings = []

        # Extract mathematical patterns from equations
        equation_patterns = {}
        for eq in latex_data['equations']:
            # Simple pattern extraction - can be enhanced
            variables = re.findall(r'[a-zA-Z](?:_\{[^}]+\})?', eq['content'])
            functions = re.findall(r'\\(?:sum|prod|int|max|min|argmax|argmin|exp|log|sqrt)', eq['content'])

            equation_patterns[eq['id']] = {
                'equation': eq,
                'variables': variables,
                'functions': functions,
                'operations': re.findall(r'[+\-*/=]', eq['content'])
            }

        # Match functions to equations
        for func in code_data['functions']:
            func_text = (func['name'] + ' ' + func['docstring'] + ' ' + func['content']).lower()

            for eq_id, eq_pattern in equation_patterns.items():
                similarity_score = 0

                # Check for mathematical function matches
                for math_func in eq_pattern['functions']:
                    if math_func.replace('\\', '') in func_text:
                        similarity_score += 2

                # Check for variable name matches
                for var in eq_pattern['variables']:
                    if var.lower() in func_text:
                        similarity_score += 1

                # Check for operation keywords
                operation_keywords = ['sum', 'product', 'integral', 'maximum', 'minimum', 'optimization']
                for keyword in operation_keywords:
                    if keyword in func_text:
                        similarity_score += 1

                # Check for algorithm-specific terms
                if 'algorithm' in func_text or 'compute' in func_text or 'calculate' in func_text:
                    similarity_score += 1

                if similarity_score >= 2:  # Threshold for relevance
                    mappings.append({
                        'equation': eq_pattern['equation'],
                        'function': func,
                        'similarity_score': similarity_score,
                        'mapping_type': 'function_to_equation'
                    })

        # Match classes to theoretical concepts
        for cls in code_data['classes']:
            class_text = (cls['name'] + ' ' + cls['docstring']).lower()

            for section in latex_data['sections']:
                if any(keyword in class_text for keyword in ['model', 'network', 'agent', 'algorithm']):
                    if any(keyword in section['content'].lower() for keyword in ['model', 'architecture', 'algorithm']):
                        mappings.append({
                            'section': section,
                            'class': cls,
                            'mapping_type': 'class_to_concept'
                        })

        return mappings

    async def build_graph(self, latex_data: Dict, code_paths: List[str]) -> LightRAG:
        """Build unified knowledge graph from LaTeX and code"""

        await self.initialize_rag()

        # Parse code files
        code_data = self.parse_code_files(code_paths)

        # Create mappings between code and equations
        mappings = self.map_code_to_equations(code_data, latex_data)

        # Prepare content for RAG insertion
        contents = []

        # Add paper sections
        for section in latex_data['sections']:
            content = f"SECTION: {section['title']}\n{section['content']}\n\n"
            if section['equations_referenced']:
                content += f"Referenced equations: {', '.join(section['equations_referenced'])}\n"
            if section['citations_referenced']:
                content += f"Citations: {', '.join(section['citations_referenced'])}\n"
            contents.append(content)

        # Add equations with context
        for eq in latex_data['equations']:
            content = f"EQUATION {eq['id']}: {eq['label']}\nMathematical expression: {eq['content']}\nLaTeX: {eq['raw_latex']}\n\n"
            contents.append(content)

        # Add code functions with mathematical context
        for func in code_data['functions']:
            content = f"FUNCTION: {func['name']} (from {func['file']})\n"
            content += f"Documentation: {func['docstring']}\n"
            content += f"Arguments: {', '.join(func['args'])}\n"
            content += f"Implementation:\n{func['content']}\n\n"
            contents.append(content)

        # Add code classes
        for cls in code_data['classes']:
            content = f"CLASS: {cls['name']} (from {cls['file']})\n"
            content += f"Documentation: {cls['docstring']}\n"
            content += f"Methods: {', '.join(cls['methods'])}\n"
            content += f"Implementation:\n{cls['content'][:1000]}...\n\n"
            contents.append(content)

        # Add explicit mappings
        for mapping in mappings:
            if mapping['mapping_type'] == 'function_to_equation':
                content = f"MAPPING: Function '{mapping['function']['name']}' implements equation {mapping['equation']['id']}\n"
                content += f"Equation: {mapping['equation']['content']}\n"
                content += f"Function: {mapping['function']['name']} in {mapping['function']['file']}\n"
                content += f"Similarity score: {mapping['similarity_score']}\n\n"
                contents.append(content)

        # Insert all content into RAG
        await self.rag.ainsert(contents)

        return self.rag

    def find_implementation_gaps(self, latex_data: Dict, code_data: Dict, mappings: List[Dict]) -> List[Dict]:
        """Identify gaps between theory and implementation"""

        gaps = []

        # Find equations without implementations
        implemented_equations = {m['equation']['id'] for m in mappings if m.get('equation')}

        for eq in latex_data['equations']:
            if eq['id'] not in implemented_equations and eq['type'] != 'inline':
                gaps.append({
                    'type': 'missing_implementation',
                    'equation': eq,
                    'description': f"Equation {eq['id']} has no corresponding implementation",
                    'priority': 'high' if 'algorithm' in eq['content'].lower() else 'medium'
                })

        # Find theoretical concepts without code
        for section in latex_data['sections']:
            if any(keyword in section['content'].lower() for keyword in ['algorithm', 'method', 'approach', 'technique']):
                has_implementation = any(
                    m['mapping_type'] == 'class_to_concept' and m['section']['title'] == section['title']
                    for m in mappings
                )

                if not has_implementation:
                    gaps.append({
                        'type': 'missing_algorithm',
                        'section': section,
                        'description': f"Section '{section['title']}' describes algorithms without implementation",
                        'priority': 'high'
                    })

        # Find code without theoretical backing
        theoretical_functions = {m['function']['name'] for m in mappings if m.get('function')}

        for func in code_data['functions']:
            if func['name'] not in theoretical_functions and not func['name'].startswith('_'):
                gaps.append({
                    'type': 'orphaned_code',
                    'function': func,
                    'description': f"Function '{func['name']}' lacks theoretical justification",
                    'priority': 'low'
                })

        return sorted(gaps, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)


async def main():
    """Example usage"""

    # Example paths - should be updated with actual files
    latex_data = {
        'title': 'Scaling Agents via Continual Pre-training',
        'sections': [
            {
                'title': 'Agentic Continual Pre-training',
                'content': 'We propose a novel approach for continual pre-training of agents...',
                'equations_referenced': ['eq_1'],
                'citations_referenced': ['smith2023']
            }
        ],
        'equations': [
            {
                'id': 'eq_1',
                'content': 'L = \\sum_{i=1}^{n} \\log p(a_i | s_i, \\theta)',
                'label': 'loss_function',
                'type': 'equation',
                'raw_latex': '\\begin{equation}L = \\sum_{i=1}^{n} \\log p(a_i | s_i, \\theta)\\end{equation}'
            }
        ]
    }

    code_paths = ["../DeepResearch-main/DeepResearch-main/inference/"]

    # Build knowledge graph
    builder = KnowledgeGraphBuilder()
    rag = await builder.build_graph(latex_data, code_paths)

    # Test query
    result = await rag.aquery("What is the loss function for agent training?", param=QueryParam(mode="hybrid"))
    print("Query result:", result)

    return builder, rag


if __name__ == "__main__":
    asyncio.run(main())