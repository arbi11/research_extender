#!/usr/bin/env python3
"""
Enhanced Chunks Generator - Improved chunk IDs for KG integration

This script:
1. Reads an existing knowledge graph JSON file
2. Extracts source files from KG entities
3. Chunks them using token-based strategy
4. Uses meaningful chunk IDs: "source_id:chunk:0"
5. Appends chunks to the knowledge graph JSON

Usage:
    python src/enhanced_chunks.py --kg-file KG_code/KG_Chp2_FieldDistribution/graph.json --output chunks_code.json
    python src/enhanced_chunks.py --kg-file KG_code/KG_LaTeX/graph.json --output chunks_latex.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import tiktoken

# EXACTLY like sample_code
DEFAULT_MAX_TOKENS = 1024
DEFAULT_OVERLAP_TOKENS = 128
TOKENIZER_MODEL = "cl100k_base"  # GPT-4 tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text - EXACTLY like sample_code"""
    return len(tokenizer.encode(text))


def chunk_by_tokens(text: str, source_id: str, chunk_order_start: int = 0,
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> List[Dict[str, Any]]:
    """
    Chunk text by token count with overlap - EXACTLY like sample_code

    ONLY CHANGE: chunk_id instead of source_chunk_index
    """
    tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)
    tokens = tokenizer.encode(text)
    chunks = []

    if len(tokens) <= max_tokens:
        # Single chunk
        chunk_id = f"{source_id}:chunk:{chunk_order_start}"
        chunks.append({
            "content": text.strip(),
            "source_id": source_id,
            "chunk_id": chunk_id  # ONLY CHANGE: meaningful ID instead of source_chunk_index
        })
    else:
        # Multiple chunks with overlap
        for i, start in enumerate(range(0, len(tokens), max_tokens - overlap_tokens)):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            chunk_id = f"{source_id}:chunk:{chunk_order_start + i}"
            chunks.append({
                "content": chunk_text.strip(),
                "source_id": source_id,
                "chunk_id": chunk_id  # ONLY CHANGE: meaningful ID instead of source_chunk_index
            })

            # Break if we've covered all tokens
            if end >= len(tokens):
                break

    return chunks


def chunk_by_lines(text: str, source_id: str, chunk_order_start: int = 0,
                   lines_per_chunk: int = 100, overlap_lines: int = 10) -> List[Dict[str, Any]]:
    """
    Simple line-based chunking (fallback strategy) - EXACTLY like sample_code

    ONLY CHANGE: chunk_id instead of source_chunk_index
    """
    lines = text.splitlines()
    chunks = []

    if len(lines) <= lines_per_chunk:
        chunk_id = f"{source_id}:chunk:{chunk_order_start}"
        chunks.append({
            "content": text.strip(),
            "source_id": source_id,
            "chunk_id": chunk_id  # ONLY CHANGE: meaningful ID instead of source_chunk_index
        })
    else:
        for i, start in enumerate(range(0, len(lines), lines_per_chunk - overlap_lines)):
            end = min(start + lines_per_chunk, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = '\n'.join(chunk_lines)

            chunk_id = f"{source_id}:chunk:{chunk_order_start + i}"
            chunks.append({
                "content": chunk_text.strip(),
                "source_id": source_id,
                "chunk_id": chunk_id  # ONLY CHANGE: meaningful ID instead of source_chunk_index
            })

            if end >= len(lines):
                break

    return chunks


def chunk_source_file(file_path: Path, source_id: str, chunk_order_start: int = 0,
                     strategy: str = "tokens") -> List[Dict[str, Any]]:
    """
    Chunk a C# source file - EXACTLY like sample_code

    ONLY CHANGE: chunk_id instead of source_chunk_index
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return []

    if not source_code.strip():
        print(f"⚠️  Empty file: {file_path}")
        return []

    # Choose chunking strategy - EXACTLY like sample_code
    if strategy == "tokens":
        return chunk_by_tokens(source_code, source_id, chunk_order_start)
    elif strategy == "lines":
        return chunk_by_lines(source_code, source_id, chunk_order_start)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def find_source_file_from_kg(kg_data: dict, source_id: str) -> str:
    """
    Find the actual file path from KG entities
    """
    for entity in kg_data.get('entities', []):
        if entity.get('source_id') == source_id:
            # Return the file_path if it exists in entity
            return entity.get('file_path', '')
    return ''


def main(kg_file: str, source_dir: str, output_chunks_file: str, strategy: str = "tokens"):
    """
    Main function to generate chunks from KG
    """
    print("🚀 Enhanced Chunks Generator - Improved chunk IDs")
    print("=" * 60)

    kg_path = Path(kg_file)
    source_base = Path(source_dir)

    if not kg_path.exists():
        print(f"❌ Error: KG file not found: {kg_file}")
        return
    
    if not source_base.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        return

    # Load KG
    print(f"📂 Loading knowledge graph from: {kg_file}")
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    print(f"   ✅ Loaded {len(kg_data.get('entities', []))} entities")
    print(f"   ✅ Loaded {len(kg_data.get('relationships', []))} relationships")

    # Extract unique source_ids from entities and build file paths
    source_file_map = {}  # source_id -> file_path
    for entity in kg_data.get('entities', []):
        source_id = entity.get('source_id')
        if source_id and source_id not in source_file_map:
            # Try both .py and .tex extensions
            py_path = source_base / f"{source_id}.py"
            tex_path = source_base / f"{source_id}.tex"
            
            if py_path.exists():
                source_file_map[source_id] = str(py_path)
            elif tex_path.exists():
                source_file_map[source_id] = str(tex_path)
            else:
                # Try without extension (for directories)
                source_file_map[source_id] = str(source_base / source_id)

    print(f"\n📊 Found {len(source_file_map)} unique source files")
    print("=" * 60)

    # Generate chunks for each source file
    all_chunks = []
    processed_files = 0
    skipped_files = 0

    for i, (source_id, file_path) in enumerate(sorted(source_file_map.items()), 1):
        print(f"[{i}/{len(source_file_map)}] Processing: {source_id}")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"   ⏭️  Source file not found: {file_path}")
            skipped_files += 1
            continue

        # Chunk the file
        chunks = chunk_source_file(file_path_obj, source_id, len(all_chunks), strategy=strategy)

        if chunks:
            all_chunks.extend(chunks)
            print(f"   ✅ Generated {len(chunks)} chunks")
            sample_id = chunks[0]['chunk_id']
            print(f"   🆔 Sample chunk ID: {sample_id}")
            processed_files += 1
        else:
            print(f"   ⚠️  No chunks generated")
            skipped_files += 1

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"✅ Processed files: {processed_files}")
    print(f"⏭️  Skipped files: {skipped_files}")
    print(f"✅ Total chunks: {len(all_chunks)}")

    # Save standalone chunks file
    chunks_data = {"chunks": all_chunks}
    with open(output_chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved chunks to: {output_chunks_file}")

    # Append to KG file
    kg_data['chunks'] = all_chunks
    with open(kg_path, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Updated KG file with chunks: {kg_file}")

    print("\n✅ Enhanced chunking complete!")

    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chunks with improved chunk IDs from KG")
    parser.add_argument("--kg-file", required=True, help="Path to KG JSON file (e.g., KG_code/KG_Chp2_FieldDistribution/graph.json)")
    parser.add_argument("--source-dir", required=True, help="Base directory containing source files (e.g., thesis_code/Chp2_MagneticFieldPredictor/magnetic_field_predictor)")
    parser.add_argument("--output", default="chunks.json", help="Output file for chunks (default: chunks.json)")
    parser.add_argument("--strategy", choices=["tokens", "lines"], default="tokens", help="Chunking strategy: tokens (default) or lines")

    args = parser.parse_args()

    chunks = main(args.kg_file, args.source_dir, args.output, args.strategy)
