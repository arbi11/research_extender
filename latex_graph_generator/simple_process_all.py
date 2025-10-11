"""
Simple processing of all LaTeX files without external APIs
"""
import asyncio
import httpx
from pathlib import Path

from .config import Config
from .processor import LatexProcessor
from .relationship_extractor import extract_mathematical_relationships

async def simple_embedding_func(texts):
    """Simple embedding function using Ollama"""
    embeddings = []
    for text in texts:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text:latest",
                    "prompt": text
                }
            )
            embedding = response.json()["embedding"]
            embeddings.append(embedding)
    return embeddings

async def build_simple_knowledge_graph():
    """Build knowledge graph without external LLM dependencies"""

    # Find all LaTeX files
    chapters_dir = Path("My_Thesis_3/My Thesis 2/Chapters")
    latex_files = list(chapters_dir.glob("*.tex"))

    print(f"Found {len(latex_files)} LaTeX files:")
    for i, file in enumerate(latex_files, 1):
        print(f"  {i}. {file.name}")

    config = Config(working_dir="./simple_thesis_kg")

    # Create simple processor (no RAG-Anything)
    processor = LatexProcessor(config)

    all_content = []
    all_relationships = []
    total_sections = 0
    total_equations = 0
    processed_files = []

    print(f"\nProcessing LaTeX files...")

    for i, latex_file in enumerate(latex_files, 1):
        print(f"\n[{i}/{len(latex_files)}] Processing: {latex_file.name}")

        try:
            # Use simple parsing directly
            parsed = processor._simple_latex_parse(str(latex_file))

            # Extract relationships
            relationships = extract_mathematical_relationships(parsed)

            # Add content
            for section in parsed.get('sections', []):
                content = f"SECTION: {section.get('title', '')}\n{section.get('content', '')}"
                all_content.append(content)

            for eq in parsed.get('equations', []):
                content = f"EQUATION {eq.get('id', '')}: {eq.get('latex', '')}"
                all_content.append(content)

            for rel in relationships:
                content = f"RELATIONSHIP ({rel['type']}): {rel}"
                all_content.append(content)

            total_sections += len(parsed.get('sections', []))
            total_equations += len(parsed.get('equations', []))
            all_relationships.extend(relationships)
            processed_files.append(latex_file.name)

            print(f"   Sections: {len(parsed.get('sections', []))}")
            print(f"   Equations: {len(parsed.get('equations', []))}")
            print(f"   Relationships: {len(relationships)}")

        except Exception as e:
            print(f"   ERROR processing {latex_file.name}: {e}")
            continue

    # Create embeddings
    print(f"\nGenerating embeddings for {len(all_content)} content items...")
    embeddings = await simple_embedding_func(all_content)

    # Save simple knowledge graph data
    output_dir = Path(config.working_dir)
    output_dir.mkdir(exist_ok=True)

    import json

    kg_data = {
        'metadata': {
            'total_files': len(processed_files),
            'processed_files': processed_files,
            'total_sections': total_sections,
            'total_equations': total_equations,
            'total_relationships': len(all_relationships),
            'total_content_items': len(all_content)
        },
        'content': all_content,
        'relationships': all_relationships,
        'embeddings': embeddings
    }

    # Save knowledge graph data
    with open(output_dir / "knowledge_graph.json", 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SIMPLE KNOWLEDGE GRAPH BUILT SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"📚 Files processed: {len(processed_files)}/{len(latex_files)}")
    print(f"📊 Total Statistics:")
    print(f"   Total Sections: {total_sections}")
    print(f"   Total Equations: {total_equations}")
    print(f"   Total Relationships: {len(all_relationships)}")
    print(f"   Content Items: {len(all_content)}")
    print(f"📁 Graph saved to: {output_dir}/knowledge_graph.json")

    # Analyze relationship types
    relationship_types = {}
    for rel in all_relationships:
        rel_type = rel.get('type', 'unknown')
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

    print(f"🔗 Relationship Types:")
    for rel_type, count in relationship_types.items():
        print(f"   {rel_type}: {count}")

    return kg_data

if __name__ == "__main__":
    asyncio.run(build_simple_knowledge_graph())