#!/usr/bin/env python3
"""
Enhanced AST to Knowledge Graph Generator

Supports both Ollama (local) and OpenRouter (cloud) LLM providers with structured outputs.

Usage:
    # OpenRouter with GPT-5, structured outputs, and source code inclusion
    python ast_to_KG.py \
        --ast-root "../../AST_code/AST_Chp2_FieldDistribution" \
        --out "../../KG_code/KG_Chp2_FieldDistribution" \
        --provider openrouter \
        --include-source-code

    # Basic usage with existing files
    python ast_to_KG.py --ast-root "../../AST_code/AST_Chp2_FieldDistribution" --out "../../KG_code/KG_Chp2_FieldDistribution"

    # Force rebuild with source code
    python ast_to_KG.py --ast-root "../../AST_code/AST_LaTeX" --out "../../KG_code/KG_LaTeX" --rebuild --include-source-code
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LLM setup
import openai

# Configuration with environment variable fallbacks
DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openrouter")
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-5")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Structured output schemas
ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "Clear description of what this code entity does"
        }
    },
    "required": ["description"],
    "additionalProperties": False
}

RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "Description of the relationship between the entities"
        },
        "weight": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence weight of this relationship (0.0-1.0)"
        }
    },
    "required": ["description", "weight"],
    "additionalProperties": False
}

@dataclass
class Entity:
    entity_name: str
    entity_type: str
    source_id: str
    description: str

@dataclass
class Relationship:
    src_id: str
    tgt_id: str
    description: str
    keywords: str
    weight: float
    source_id: str

class LLMClient:
    """Unified LLM client supporting both Ollama and OpenRouter"""

    def __init__(self, provider: str = "openrouter", use_structured: bool = True, include_source_code: bool = False):
        self.provider = provider
        self.use_structured = use_structured
        self.include_source_code = include_source_code

        if provider == "openrouter":
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter provider")
            self.client = openai.OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
            self.model = DEFAULT_MODEL
        elif provider == "ollama":
            self.client = openai.OpenAI(
                base_url=OLLAMA_HOST + "/v1",
                api_key="ollama"  # Ollama doesn't need a real key
            )
            self.model = OLLAMA_MODEL
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_entity_description(self, symbol: dict, relative_path: str, source_code: str = "") -> str:
        """Get structured entity description with optional source code"""
        # Build enhanced prompt with source code if available
        if self.include_source_code and source_code:
            prompt = f"""Describe this code entity from file {relative_path}:

Entity: {symbol.get('name', 'Unknown')}
Source Code:
{source_code}

Additional metadata:
{json.dumps({k: v for k, v in symbol.items() if k not in ['name', 'source_code']}, indent=2)}

Provide a clear, concise description of what this entity does based on the actual implementation."""
        else:
            prompt = f"""Describe this code entity from file {relative_path}:

{json.dumps(symbol, indent=2)}

Provide a clear, concise description of what this entity does."""

        if self.provider == "openrouter" and self.use_structured:
            # Use structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "entity_description",
                        "strict": True,
                        "schema": ENTITY_SCHEMA
                    }
                }
            )
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result["description"]
        else:
            # Use regular text response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

    def get_relationship_description(self, symbol: dict, src_entity: str, tgt_entity: str, rel_type: str) -> Tuple[str, float]:
        """Get structured relationship description with weight"""
        prompt = f"""Describe this {rel_type} relationship:

Source: {src_entity}
Target: {tgt_entity}

Reference data:
{json.dumps(symbol, indent=2)}

Provide a clear description and a confidence weight (0.0-1.0)."""

        if self.provider == "openrouter" and self.use_structured:
            # Use structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "relationship_description",
                        "strict": True,
                        "schema": RELATIONSHIP_SCHEMA
                    }
                }
            )
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result["description"], result["weight"]
        else:
            # Use regular text response with parsing
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()

            # Try to extract weight from response

            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            weight = float(numbers[-1]) if numbers else 0.5
            weight = max(0.0, min(1.0, weight))  # Clamp to 0-1

            return content, weight

def check_existing_output(output_dir: Path) -> bool:
    """Check if output files already exist"""
    entities_file = output_dir / "entities.json"
    relationships_file = output_dir / "relationships.json"
    return entities_file.exists() and relationships_file.exists()

def extract_source_code(file_path: str, start_line: int, end_line: int) -> str:
    """Extract source code lines from a file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Convert to 0-based indexing and clamp to valid range
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)

    if start_idx >= len(lines) or start_idx >= end_idx:
        return ""

    # Extract the lines
    extracted_lines = lines[start_idx:end_idx]

    # Remove common leading whitespace for cleaner presentation
    if extracted_lines:
        # Find minimum indentation (excluding empty lines)
        non_empty_lines = [line for line in extracted_lines if line.strip()]
        if non_empty_lines:
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
            extracted_lines = [line[min_indent:] if len(line) > min_indent else line for line in extracted_lines]

    return ''.join(extracted_lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate Knowledge Graph from AST JSON files with resume capability")
    parser.add_argument("--ast-root", default="AST_code/AST_Chp2_FieldDistribution",
                       help="Root directory containing AST JSON files (default: AST_code/AST_Chp2_FieldDistribution)")
    parser.add_argument("--out", default="KG_code/KG_Chp2_FieldDistribution",
                       help="Output directory for KG files (default: KG_code/KG_Chp2_FieldDistribution)")
    parser.add_argument("--graph-name", default="graph.json",
                       help="Output graph filename (default: graph.json)")
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=DEFAULT_PROVIDER,
                       help=f"LLM provider (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--structured-outputs", action="store_true", default=True,
                       help="Use structured outputs for cleaner responses (default: True)")
    parser.add_argument("--include-source-code", action="store_true", default=False,
                       help="Include actual source code lines in LLM prompts for better descriptions")
    parser.add_argument("--rebuild", action="store_true", default=False,
                       help="Rebuild from scratch, ignoring existing output files")

    args = parser.parse_args()

    # Initialize LLM client
    llm_client = LLMClient(
        provider=args.provider,
        use_structured=args.structured_outputs,
        include_source_code=args.include_source_code
    )

    pAST_dir = Path(args.ast_root)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing output
    if not args.rebuild and check_existing_output(output_dir):
        print("📂 Existing knowledge graph found. Use --rebuild to regenerate.")
        print(f"📁 Output directory: {output_dir}")
        return

    print("🚀 Enhanced AST to KG")
    print(f"📁 AST root: {args.ast_root}")
    print(f"📁 Output dir: {args.out}")
    print(f"🤖 LLM: {args.provider} + {llm_client.model} + structured_outputs={args.structured_outputs}")

    all_entities = {}
    all_relationships = []

    # Create namespace mapping: script_relative_path -> [namespaces]
    namespace_map = {}

    # Phase 1: Entities from definitions
    for ast_file in pAST_dir.rglob("*.json"):
        print('*'*50)
        print('ANALYZING:', ast_file)
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_json = json.load(f)

        relative_path = str(ast_file.relative_to(pAST_dir)).replace('.json', '').replace('\\', '/')
        print(f"Processing {relative_path}...")

        # Get the original source file path directly from AST metadata
        original_file_path = ast_json.get('file_path', '')
        if args.include_source_code and original_file_path:
            if not Path(original_file_path).exists():
                print(f"⚠️  Source file not found: {original_file_path}")
                original_file_path = None

        for symbol in ast_json['definitions']:
            entity_name = symbol['name']
            print('ENTITY:', entity_name)

            # Build namespace mapping
            if symbol['type'] == 'namespaces':
                if relative_path not in namespace_map:
                    namespace_map[relative_path] = []
                namespace_name = entity_name.split('::')[-1]
                if namespace_name not in namespace_map[relative_path]:
                    namespace_map[relative_path].append(namespace_name)

            if entity_name not in all_entities:
                # Extract source code if enabled and available
                source_code = ""
                if args.include_source_code and original_file_path:
                    start_line = symbol.get('start_line', 0)
                    end_line = symbol.get('end_line', 0)
                    if start_line and end_line:
                        source_code = extract_source_code(original_file_path, start_line, end_line)
                        if source_code:
                            print(f"📝 Extracted {len(source_code)} characters of source code")

                print('SYMBOL:', symbol)
                description = llm_client.get_entity_description(symbol, relative_path, source_code)
                print('DESCRIPTION:', description)
                all_entities[entity_name] = Entity(
                    entity_name=entity_name,
                    entity_type=symbol['type'],
                    source_id=relative_path,
                    description=description
                )
            print('-'*50)

    print('NAMESPACE_MAP:', namespace_map)

    print('='*50)
    print('+'*50)
    print('='*50)

    # Phase 2: Relationships from references
    for ast_file in pAST_dir.rglob("*.json"):
        print('*'*50)
        print('ANALYZING:', ast_file)
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_json = json.load(f)

        relative_path = str(ast_file.relative_to(pAST_dir)).replace('.json', '').replace('\\', '/')
        print(f"Processing {relative_path}...")

        for symbol in ast_json['references']:
            source_entity = symbol['calling_entity']
            target_entity = symbol['name'].replace('.', '::')

            # Resolve target_entity using namespace mapping
            if target_entity not in all_entities:
                for file_path, namespaces in namespace_map.items():
                    for namespace in namespaces:
                        if target_entity.startswith(f"{namespace}::"):
                            target_entity = f"{file_path}::{target_entity}"
                            break
                    if target_entity in all_entities:
                        break

            # Skip external dependencies
            if target_entity not in all_entities:
                print(f"SKIPPING external dependency: {target_entity}")
                continue

            print('SOURCE_ENTITY:', source_entity, 'YES' if source_entity in all_entities.keys() else 'NO')
            print('TARGET_ENTITY:', target_entity, 'YES' if target_entity in all_entities.keys() else 'NO')
            print('SYMBOL:', symbol)

            description, weight = llm_client.get_relationship_description(symbol, source_entity, target_entity, symbol['type'])
            print('DESCRIPTION:', description)
            print('WEIGHT:', weight)

            all_relationships.append(Relationship(
                src_id=source_entity,
                tgt_id=target_entity,
                description=description,
                keywords=symbol['type'],
                weight=weight,
                source_id=relative_path
            ))
            print('-'*50)

    # Results
    knowledge_graph = {
        "entities": [
            {
                "entity_name": e.entity_name,
                "entity_type": e.entity_type,
                "description": e.description,
                "source_id": e.source_id
            }
            for e in all_entities.values()
        ],
        "relationships": [
            {
                "src_id": r.src_id,
                "tgt_id": r.tgt_id,
                "description": r.description,
                "keywords": r.keywords,
                "weight": r.weight,
                "source_id": r.source_id
            }
            for r in all_relationships
        ]
    }

    print(f"✅ Created {len(knowledge_graph['entities'])} entities")
    print(f"✅ Created {len(knowledge_graph['relationships'])} relationships")

    # Save separate files for entities and relationships
    entities_file = output_dir / "entities.json"
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph['entities'], f, indent=2, ensure_ascii=False)
    print(f"💾 Saved entities: {entities_file}")

    relationships_file = output_dir / "relationships.json"
    with open(relationships_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph['relationships'], f, indent=2, ensure_ascii=False)
    print(f"💾 Saved relationships: {relationships_file}")

    # Save combined graph
    output_file = output_dir / args.graph_name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved combined graph: {output_file}")

    # Print sample of results
    if knowledge_graph['entities']:
        print("\n📋 Sample entities:")
        for entity in knowledge_graph['entities'][:3]:
            print(f"  • {entity['entity_name']}: {entity['description'][:100]}...")

    if knowledge_graph['relationships']:
        print("\n🔗 Sample relationships:")
        for rel in knowledge_graph['relationships'][:3]:
            print(f"  • {rel['src_id']} → {rel['tgt_id']} (weight: {rel['weight']})")

if __name__ == "__main__":
    main()