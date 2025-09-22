import ast
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import re

from dotenv import load_dotenv
import yaml

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()

# Load YAML configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

SUPPORTED_CODE_GLOBS = CONFIG.get("code_files", {}).get(
    "supported_extensions",
    ["*.py", "*.cpp", "*.c", "*.h", "*.js", "*.ts", "*.java"],
)


def _collect_code_files(paths: List[str]) -> List[str]:
    """Expand any directories into code file lists using SUPPORTED_CODE_GLOBS."""
    files: List[str] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            files.append(str(path))
        elif path.is_dir():
            for pat in SUPPORTED_CODE_GLOBS:
                files.extend([str(f) for f in path.rglob(pat)])
        # silently ignore non-existent
    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for f in files:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


class KnowledgeGraphBuilder:
    """Build unified LightRAG store from LaTeX dict and code files, using OpenRouter as LLM."""

    def __init__(
        self,
        working_dir: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ):
        # Resolve config-driven defaults
        dirs_cfg = CONFIG.get("directories", {})
        llm_cfg = CONFIG.get("llm", {})
        emb_cfg = CONFIG.get("embedding", {})
        proc_cfg = CONFIG.get("processing", {})

        self.working_dir = Path(working_dir or dirs_cfg.get("working_dir", "./research_graph"))
        self.llm_model = llm_model or llm_cfg.get("model")

        # OpenRouter config
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", llm_cfg.get("base_url", "https://openrouter.ai/api/v1"))

        # LLM defaults
        self.llm_temperature = llm_cfg.get("temperature", 0)
        self.llm_max_tokens = llm_cfg.get("max_tokens", 4000)

        # Embeddings
        self.embedding_model_name = embedding_model or emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = embedding_dim or emb_cfg.get("dimension", 384)
        self.embedding_max_token_size = emb_cfg.get("max_token_size", 5000)
        self.embed_model = SentenceTransformer(self.embedding_model_name)

        # Processing thresholds
        self.similarity_threshold = proc_cfg.get("similarity_threshold", 2)
        self.gap_priority_map = proc_cfg.get("gap_priorities", {"high": 3, "medium": 2, "low": 1})

        # FS
        os.makedirs(self.working_dir, exist_ok=True)

        # LightRAG instance (lazy)
        self.rag: LightRAG | None = None

    async def llm_model_func(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        """LLM function for LightRAG via OpenRouter (OpenAI-compatible)."""
        if not self.openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment")

        client = AsyncOpenAI(
            api_key=self.openrouter_key,
            base_url=self.openrouter_base,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            # Expecting list of dicts like {"role": "...", "content": "..."}
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=kwargs.get("temperature", getattr(self, "llm_temperature", 0)),
            max_tokens=kwargs.get("max_tokens", getattr(self, "llm_max_tokens", 4000)),
        )
        return response.choices[0].message.content

    async def initialize_rag(self):
        """Initialize the LightRAG system with SentenceTransformer embeddings."""
        async def _embed(texts: List[str]) -> List[List[float]]:
            # SentenceTransformer returns numpy array; convert to Python lists
            embs = self.embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            if isinstance(embs, np.ndarray):
                return embs.tolist()
            # Fallback in case a single vector is returned
            return [list(map(float, v)) for v in embs]

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.llm_model_func,
            llm_model_name=self.llm_model,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=_embed,
            ),
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    def parse_code_files(self, code_paths: List[str]) -> Dict[str, Any]:
        """Parse code files and extract structure (functions, classes, imports)."""
        code_data: Dict[str, Any] = {
            "files": [],
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "algorithms": [],
        }

        for code_path in code_paths:
            file_path = Path(code_path)
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # If read fails, skip file
                continue

            file_info = {
                "path": str(file_path),
                "name": file_path.name,
                "content": content,
                "size": len(content),
            }

            if file_path.suffix == ".py":
                try:
                    tree = ast.parse(content)
                except Exception:
                    tree = None

                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_info = {
                                "name": node.name,
                                "file": str(file_path),
                                "docstring": ast.get_docstring(node) or "",
                                "args": [arg.arg for arg in node.args.args],
                                "line_number": node.lineno,
                                "content": ast.get_source_segment(content, node) or "",
                            }
                            code_data["functions"].append(func_info)

                        elif isinstance(node, ast.ClassDef):
                            class_info = {
                                "name": node.name,
                                "file": str(file_path),
                                "docstring": ast.get_docstring(node) or "",
                                "line_number": node.lineno,
                                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                                "content": ast.get_source_segment(content, node) or "",
                            }
                            code_data["classes"].append(class_info)

                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                code_data["imports"].append(
                                    {"module": alias.name, "alias": alias.asname, "file": str(file_path)}
                                )

                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for alias in node.names:
                                code_data["imports"].append(
                                    {"module": f"{module}.{alias.name}", "alias": alias.asname, "file": str(file_path)}
                                )

            code_data["files"].append(file_info)

        return code_data

    def map_code_to_equations(self, code_data: Dict, latex_data: Dict) -> List[Dict]:
        """Heuristically map code implementations to mathematical equations and concepts."""
        mappings: List[Dict] = []

        # Extract mathematical patterns from equations
        equation_patterns: Dict[str, Any] = {}
        for eq in latex_data.get("equations", []):
            variables = re.findall(r"[a-zA-Z](?:_\{[^}]+\})?", eq.get("content", ""))
            functions = re.findall(r"\\(?:sum|prod|int|max|min|argmax|argmin|exp|log|sqrt)", eq.get("content", ""))

            equation_patterns[eq["id"]] = {
                "equation": eq,
                "variables": variables,
                "functions": functions,
                "operations": re.findall(r"[+\-*/=]", eq.get("content", "")),
            }

        # Match functions to equations
        for func in code_data.get("functions", []):
            func_text = (func["name"] + " " + func["docstring"] + " " + func["content"]).lower()

            for eq_id, eq_pattern in equation_patterns.items():
                similarity_score = 0

                for math_func in eq_pattern["functions"]:
                    if math_func.replace("\\", "") in func_text:
                        similarity_score += 2

                for var in eq_pattern["variables"]:
                    if var.lower() in func_text:
                        similarity_score += 1

                operation_keywords = ["sum", "product", "integral", "maximum", "minimum", "optimization"]
                for keyword in operation_keywords:
                    if keyword in func_text:
                        similarity_score += 1

                if any(k in func_text for k in ["algorithm", "compute", "calculate"]):
                    similarity_score += 1

                if similarity_score >= self.similarity_threshold:
                    mappings.append(
                        {
                            "equation": eq_pattern["equation"],
                            "function": func,
                            "similarity_score": similarity_score,
                            "mapping_type": "function_to_equation",
                        }
                    )

        # Match classes to theoretical concepts
        for cls in code_data.get("classes", []):
            class_text = (cls["name"] + " " + cls["docstring"]).lower()
            for section in latex_data.get("sections", []):
                if any(k in class_text for k in ["model", "network", "agent", "algorithm"]):
                    if any(k in section.get("content", "").lower() for k in ["model", "architecture", "algorithm"]):
                        mappings.append({"section": section, "class": cls, "mapping_type": "class_to_concept"})

        return mappings

    async def build_graph(self, latex_data: Dict, code_paths: List[str]) -> LightRAG:
        """Prepare and insert content into LightRAG only (no explicit networkx graph)."""

        await self.initialize_rag()

        # Expand directories to files
        code_files = _collect_code_files(code_paths)

        # Parse code files
        code_data = self.parse_code_files(code_files)

        # Create mappings between code and equations
        mappings = self.map_code_to_equations(code_data, latex_data)

        # Prepare content for RAG insertion
        contents: List[str] = []

        # Add paper sections
        for section in latex_data.get("sections", []):
            content = f"SECTION: {section.get('title','')}\n{section.get('content','')}\n\n"
            if section.get("equations_referenced"):
                content += f"Referenced equations: {', '.join(section['equations_referenced'])}\n"
            if section.get("citations_referenced"):
                content += f"Citations: {', '.join(section['citations_referenced'])}\n"
            contents.append(content)

        # Add equations with context
        for eq in latex_data.get("equations", []):
            content = (
                f"EQUATION {eq.get('id','')}: {eq.get('label','')}\n"
                f"Mathematical expression: {eq.get('content','')}\n"
                f"LaTeX: {eq.get('raw_latex','')}\n\n"
            )
            contents.append(content)

        # Add code functions with mathematical context
        for func in code_data.get("functions", []):
            content = (
                f"FUNCTION: {func['name']} (from {func['file']})\n"
                f"Documentation: {func['docstring']}\n"
                f"Arguments: {', '.join(func['args'])}\n"
                f"Implementation:\n{func['content']}\n\n"
            )
            contents.append(content)

        # Add code classes
        for cls in code_data.get("classes", []):
            content = (
                f"CLASS: {cls['name']} (from {cls['file']})\n"
                f"Documentation: {cls['docstring']}\n"
                f"Methods: {', '.join(cls['methods'])}\n"
                f"Implementation:\n{cls['content'][:1000]}...\n\n"
            )
            contents.append(content)

        # Add explicit mappings
        for mapping in mappings:
            if mapping["mapping_type"] == "function_to_equation":
                content = (
                    f"MAPPING: Function '{mapping['function']['name']}' implements equation {mapping['equation']['id']}\n"
                    f"Equation: {mapping['equation']['content']}\n"
                    f"Function: {mapping['function']['name']} in {mapping['function']['file']}\n"
                    f"Similarity score: {mapping['similarity_score']}\n\n"
                )
                contents.append(content)

        # Insert all content into RAG
        await self.rag.ainsert(contents)

        return self.rag

    def find_implementation_gaps(self, latex_data: Dict, code_data: Dict, mappings: List[Dict]) -> List[Dict]:
        """Identify gaps between theory and implementation."""
        gaps: List[Dict] = []

        # Equations without implementations
        implemented_equations = {m["equation"]["id"] for m in mappings if m.get("equation")}
        for eq in latex_data.get("equations", []):
            if eq["id"] not in implemented_equations and eq.get("type") != "inline":
                gaps.append(
                    {
                        "type": "missing_implementation",
                        "equation": eq,
                        "description": f"Equation {eq['id']} has no corresponding implementation",
                        "priority": "high" if "algorithm" in eq.get("content", "").lower() else "medium",
                    }
                )

        # Theoretical concepts without code
        for section in latex_data.get("sections", []):
            if any(k in section.get("content", "").lower() for k in ["algorithm", "method", "approach", "technique"]):
                has_impl = any(
                    m["mapping_type"] == "class_to_concept" and m["section"]["title"] == section["title"] for m in mappings
                )
                if not has_impl:
                    gaps.append(
                        {
                            "type": "missing_algorithm",
                            "section": section,
                            "description": f"Section '{section['title']}' describes algorithms without implementation",
                            "priority": "high",
                        }
                    )

        # Code without theoretical backing
        theoretical_functions = {m["function"]["name"] for m in mappings if m.get("function")}
        for func in code_data.get("functions", []):
            if func["name"] not in theoretical_functions and not func["name"].startswith("_"):
                gaps.append(
                    {
                        "type": "orphaned_code",
                        "function": func,
                        "description": f"Function '{func['name']}' lacks theoretical justification",
                        "priority": "low",
                    }
                )

        return sorted(gaps, key=lambda x: self.gap_priority_map.get(x["priority"], 0), reverse=True)


async def main():
    """Example usage (uses synthetic LaTeX dict and expands code dirs to files)."""

    # Example LaTeX-like data structure; in production, use latex_processor.process_latex(.tex)
    latex_data = {
        "title": "Scaling Agents via Continual Pre-training",
        "sections": [
            {
                "title": "Agentic Continual Pre-training",
                "content": "We propose a novel approach for continual pre-training of agents...",
                "equations_referenced": ["eq_1"],
                "citations_referenced": ["smith2023"],
            }
        ],
        "equations": [
            {
                "id": "eq_1",
                "content": "L = \\sum_{i=1}^{n} \\log p(a_i | s_i, \\theta)",
                "label": "loss_function",
                "type": "equation",
                "raw_latex": "\\begin{equation}L = \\sum_{i=1}^{n} \\log p(a_i | s_i, \\theta)\\end{equation}",
            }
        ],
    }

    # Provide file paths or directories; directories will be expanded safely
    code_paths = ["./src"]  # for demo, expand src dir; replace with your actual repo paths

    builder = KnowledgeGraphBuilder()

    rag = await builder.build_graph(latex_data, code_paths)

    # Test query
    result = await rag.aquery("What is the loss function for agent training?", param=QueryParam(mode="hybrid"))
    print("Query result:", result)

    return builder, rag


if __name__ == "__main__":
    asyncio.run(main())
