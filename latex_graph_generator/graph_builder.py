import os
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from .processor import LatexProcessor
from .config import Config

class GraphBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.processor = LatexProcessor(config)
        self.lightrag = None

    async def build(self, latex_file: str, llm_func, embedding_func):
        os.makedirs(self.config.working_dir, exist_ok=True)

        self.lightrag = LightRAG(
            working_dir=self.config.working_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )

        await self.lightrag.initialize_storages()
        await initialize_pipeline_status()

        # Process LaTeX
        processed = await self.processor.process(latex_file, self.lightrag)

        # Insert into graph
        content_chunks = []

        # Add sections
        for section in processed['sections']:
            chunk = f"SECTION: {section.get('title', '')}\n{section.get('content', '')}"
            content_chunks.append(chunk)

        # Add equations with mathematical context
        for eq in processed['equations']:
            chunk = f"EQUATION {eq.get('id', '')}: {eq.get('label', '')}\n"
            chunk += f"LaTeX: {eq.get('latex', '')}\n"
            chunk += f"Content: {eq.get('text', '')}"
            content_chunks.append(chunk)

        # Add mathematical relationships
        for rel in processed['relationships']:
            chunk = f"MATHEMATICAL RELATIONSHIP ({rel['type']}):\n"
            chunk += f"Details: {rel}"
            content_chunks.append(chunk)

        await self.lightrag.ainsert(content_chunks)
        await self.lightrag.finalize_storages()

        return {
            'total_sections': len(processed['sections']),
            'total_equations': len(processed['equations']),
            'total_relationships': len(processed['relationships']),
            'graph_directory': self.config.working_dir,
            'relationship_types': list(set(rel['type'] for rel in processed['relationships']))
        }

    async def query(self, query: str, mode: str = "hybrid"):
        if not self.lightrag:
            raise ValueError("Knowledge graph not built yet")

        from lightrag import QueryParam
        result = await self.lightrag.aquery(query, param=QueryParam(mode=mode))
        return result