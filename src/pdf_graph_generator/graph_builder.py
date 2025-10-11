import os
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from .relationship_extractor import extract_mathematical_relationships

class GraphBuilder:
    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir or os.getenv("PDF_KG_WORKING_DIR", "./pdf_kg_output")
        self.parser = os.getenv("PDF_KG_PARSER", "mineru")
        self.enable_math_analysis = os.getenv("PDF_KG_ENABLE_MATH", "true").lower() == "true"
        self.lightrag = None
        self.llm_func = None
        self.embedding_func = None

    async def _vision_model_func(self, prompt, system_prompt=None, image_data=None, **kwargs):
        """Vision model function for processing images with qwen3-vl-30b-a3b-thinking"""
        if not image_data:
            # Fallback to regular text processing when no image
            if self.llm_func:
                return await self.llm_func(prompt, system_prompt, **kwargs)
            return "No image data provided"

        # Use OpenRouter API for vision model
        import httpx

        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("VISION_MODEL", "google/gemini-2.5-flash-lite")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Construct message with image
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        ]
        messages.append({"role": "user", "content": content})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages
                },
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    async def _process_pdf(self, pdf_file: str):
        """Process PDF document using RAGAnything"""
        from raganything import RAGAnything, RAGAnythingConfig

        rag_config = RAGAnythingConfig(
            parser=self.parser,
            enable_equation_processing=self.enable_math_analysis,
            enable_table_processing=False,
            enable_image_processing=True,
        )

        rag_anything = RAGAnything(
            lightrag=self.lightrag,
            config=rag_config,
            llm_model_func=self.llm_func,
            embedding_func=self.embedding_func,
            vision_model_func=self._vision_model_func,
        )

        # Parse document
        parsed = await rag_anything.process_document_complete(pdf_file)

        # Extract mathematical relationships
        relationships = extract_mathematical_relationships(parsed)

        return {
            'sections': parsed.get('sections', []),
            'equations': parsed.get('equations', []),
            'relationships': relationships,
            'raw_parsed': parsed
        }

    async def build(self, pdf_file: str, llm_func, embedding_func):
        os.makedirs(self.working_dir, exist_ok=True)

        # Store the function references for RAGAnything
        self.llm_func = llm_func
        self.embedding_func = embedding_func

        self.lightrag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )

        await self.lightrag.initialize_storages()
        await initialize_pipeline_status()

        # Process PDF document
        processed = await self._process_pdf(pdf_file)

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
            'graph_directory': self.working_dir,
            'relationship_types': list(set(rel['type'] for rel in processed['relationships']))
        }

    async def query(self, query: str, mode: str = "hybrid"):
        if not self.lightrag:
            raise ValueError("Knowledge graph not built yet")

        from lightrag import QueryParam
        result = await self.lightrag.aquery(query, param=QueryParam(mode=mode))
        return result