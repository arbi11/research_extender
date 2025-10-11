import re
import asyncio
from pathlib import Path
from .relationship_extractor import extract_mathematical_relationships
from .config import Config

class LatexProcessor:
    def __init__(self, config: Config):
        self.config = config

    async def process(self, latex_file: str, lightrag_instance):
        from raganything import RAGAnything, RAGAnythingConfig

        rag_config = RAGAnythingConfig(
            parser=self.config.parser,
            enable_equation_processing=self.config.enable_math_analysis,
            enable_table_processing=True,
            enable_image_processing=False,
        )

        rag_anything = RAGAnything(
            lightrag=lightrag_instance,
            config=rag_config,
        )

        # Parse document
        parsed = await rag_anything.process_document_complete(latex_file)

        # Extract mathematical relationships
        relationships = extract_mathematical_relationships(parsed)

        return {
            'sections': parsed.get('sections', []),
            'equations': parsed.get('equations', []),
            'relationships': relationships,
            'raw_parsed': parsed
        }

    def _simple_latex_parse(self, latex_file: str):
        """Simple LaTeX parsing fallback"""
        content = Path(latex_file).read_text(encoding='utf-8')

        # Extract sections
        section_pattern = r'\\(?:section|subsection|subsubsection|chapter)\*?\{([^}]+)\}'
        sections = []
        for match in re.finditer(section_pattern, content):
            sections.append({
                'title': match.group(1),
                'content': self._extract_section_content(content, match.start()),
                'type': 'section'
            })

        # Extract equations
        eq_patterns = [
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\begin\{align\}(.*?)\\end\{align\}',
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            r'\$\$(.*?)\$\$'
        ]

        equations = []
        eq_id = 1
        for pattern in eq_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                equations.append({
                    'id': f'eq_{eq_id}',
                    'latex': match.group(1).strip(),
                    'text': match.group(1).strip()
                })
                eq_id += 1

        return {
            'sections': sections,
            'equations': equations,
            'content': content
        }

    def _extract_section_content(self, content: str, start_pos: int):
        """Extract content until next section"""
        remaining = content[start_pos:]
        next_section = re.search(r'\\(?:section|subsection|subsubsection|chapter)\*?\{', remaining)
        if next_section:
            return remaining[:next_section.start()]
        return remaining[:1000]  # Limit content length