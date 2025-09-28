#!/usr/bin/env python3
"""
Simple AST to Knowledge Graph Generator

Ultra simple approach:
- Definitions → Entities (dump full AST data to LLM)
- References → Relationships (dump full AST data to LLM)
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# LLM setup
import openai
import json
from typing import Tuple

# Configuration
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "gemma3:4b"

client = openai.OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

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

def get_entity_description(symbol: dict, relative_path: str) -> str:
    """Get LLM description - just dump the whole symbol"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Describe this code entity from file {relative_path}:

{json.dumps(symbol, indent=2)}

Provide a clear description of what this entity does."""
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"Function or class definition in {relative_path}"

def get_relationship_description(symbol: dict, src_entity: str, tgt_entity: str, rel_type: str) -> Tuple[str, float]:
    """Get LLM description - just dump the whole symbol"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Describe this {rel_type} relationship:

Source: {src_entity}
Target: {tgt_entity}

Reference data:
{json.dumps(symbol, indent=2)}

Provide description and weight (0.0-1.0)."""
            }]
        )
        content = response.choices[0].message.content.strip()

        # Try to extract weight from response
        try:
            # Look for a number in the response
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            weight = float(numbers[-1]) if numbers else 0.5
            weight = max(0.0, min(1.0, weight))  # Clamp to 0-1
        except:
            weight = 0.5

        return content, weight
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"{rel_type} relationship between {src_entity} and {tgt_entity}", 0.5

def main():
    parser = argparse.ArgumentParser(description="Generate Knowledge Graph from AST JSON files")
    parser.add_argument("--ast-root", default="AST_code/AST_Chp2_FieldDistribution",
                       help="Root directory containing AST JSON files (default: AST_code/AST_Chp2_FieldDistribution)")
    parser.add_argument("--out", default="KG_code/KG_Chp2_FieldDistribution",
                       help="Output directory for KG files (default: KG_code/KG_Chp2_FieldDistribution)")
    parser.add_argument("--graph-name", default="graph.json",
                       help="Output graph filename (default: graph.json)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama host URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="gemma3:4b",
                       help="LLM model name (default: gemma3:4b)")

    args = parser.parse_args()

    # Update configuration from CLI
    global OLLAMA_HOST, LLM_MODEL
    OLLAMA_HOST = args.ollama_host
    LLM_MODEL = args.model

    # Recreate client with updated config
    client = openai.OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

    print("🚀 Simple AST to KG")
    print(f"📁 AST root: {args.ast_root}")
    print(f"📁 Output dir: {args.out}")
    print(f"🤖 LLM: {OLLAMA_HOST} + {LLM_MODEL}")

    pAST_dir = Path(args.ast_root)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        for symbol in ast_json['definitions']:
            # entity_name = f"{relative_path}::{symbol['name']}"
            entity_name = symbol['name']
            print('ENTITY:', entity_name)

            # Build namespace mapping
            if symbol['type'] == 'namespaces':
                if relative_path not in namespace_map:
                    namespace_map[relative_path] = []
                # Extract namespace name from hierarchical name (e.g., "TestMain::MainTestNamespace" -> "MainTestNamespace")
                namespace_name = entity_name.split('::')[-1]
                if namespace_name not in namespace_map[relative_path]:
                    namespace_map[relative_path].append(namespace_name)

            if entity_name not in all_entities:
                print('SYMBOL:', symbol)
                description = get_entity_description(symbol, relative_path)
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
                # Try to find the correct file for this target
                for file_path, namespaces in namespace_map.items():
                    for namespace in namespaces:
                        if target_entity.startswith(f"{namespace}::"):
                            # Construct full target entity name
                            target_entity = f"{file_path}::{target_entity}"
                            break
                    if target_entity in all_entities:
                        break

            # Skip external dependencies - only include relationships within our codebase
            if target_entity not in all_entities:
                print(f"SKIPPING external dependency: {target_entity}")
                continue

            print('SOURCE_ENTITY:', source_entity, 'YES' if source_entity in all_entities.keys() else 'NO')
            print('TARGET_ENTITY:', target_entity, 'YES' if target_entity in all_entities.keys() else 'NO')
            print('SYMBOL:', symbol)

            description, weight = get_relationship_description(symbol, source_entity, target_entity, symbol['type'])
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

    output_file = output_dir / args.graph_name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    main()
