# Enhanced LightRAG Integration System

A complete system for building and querying knowledge graphs from LaTeX papers and source code using LightRAG with OpenRouter LLMs and Ollama embeddings.

## Architecture Overview

### Two-KG System
- **Code KG**: Merged knowledge graph from all thesis code chapters (Chp2-5)
- **LaTeX KG**: Knowledge graph from all LaTeX thesis chapters
- **3-Response Query**: Each query gets 3 responses (Code, LaTeX, Combined)

## Complete Pipeline

### Phase 1: Generate AST Files

```bash
# Python Code - All Chapters (Monolithic AST)
python src/graph_generator/extract_KG_python.py thesis_code/ --output-folder AST_code/AST_code_ALL --no-copy-source

# LaTeX - All Chapters
python src/graph_generator/latex_code_extractor.py thesis_latex/Chapters --output-folder AST_code/AST_LaTeX --pattern "*.tex" --no-copy-source
```

### Phase 2: Generate Knowledge Graphs

```bash
# Code KG - All Chapters Combined (Monolithic KG)
python src/graph_generator/ast_to_KG.py --ast-root AST_code/AST_code_ALL --out KG_code/KG_code_ALL --provider openrouter --include-source-code

# LaTeX KG
python src/graph_generator/ast_to_KG.py --ast-root AST_code/AST_LaTeX --out KG_code/KG_LaTeX --provider openrouter --include-source-code
```

### Phase 3: Generate Chunks

```bash
# Code KG
python src/enhanced_chunks.py --kg-file KG_code/KG_code_ALL/graph.json --source-dir thesis_code --output chunks_code_all.json

# LaTeX KG
python src/enhanced_chunks.py --kg-file KG_code/KG_LaTeX/graph.json --source-dir thesis_latex/Chapters --output chunks_latex.json

```

### Phase 5: Integrate with LightRAG (Separate Indices)

```bash
# Code KG → lightrag_code_index
python src/enhanced_integrate_extKG.py --kg-file KG_code/KG_code_ALL/graph.json --output-dir ./lightrag_code_index

# LaTeX KG → lightrag_latex_index
python src/enhanced_integrate_extKG.py --kg-file KG_code/KG_LaTeX/graph.json --output-dir ./lightrag_latex_index


```

### Phase 6: Query with 3-Response System

```bash
# Interactive query interface
python src/enhanced_interface.py
```

**Query Flow:**
1. Select LLM model (Claude/Gemini/GPT-5/Custom)
2. Enter your question
3. Get 3 responses:
   - **Response 1**: From Code KG (implementation details)
   - **Response 2**: From LaTeX KG (theory/equations)
   - **Response 3**: Combined synthesis

## Enhanced Components

### `enhanced_chunks.py`
Generates chunks with meaningful IDs:
- Uses tiktoken with "cl100k_base" tokenizer (GPT-4)
- Creates traceable chunk IDs: `"main:chunk:0"`, `"src/model.py:chunk:1"`
- Supports token-based and line-based chunking
- Requires `--source-dir` to locate actual source files

### `enhanced_integrate_extKG.py`
Integrates KG files with LightRAG:
- **LLM**: OpenRouter (any model: Claude, Gemini, GPT-5, custom)
- **Embeddings**: Ollama localhost (`bge-m3:latest`, 1024 dim)
- **NEW**: `--output-dir` parameter for separate indices
- Uses `openai_complete_if_cache` for OpenRouter compatibility

### `enhanced_interface.py`
Interactive querying with 3-response system:
- **Model Selection**: 4 options with custom model support
- **Dual-KG Queries**: Separate Code and LaTeX responses
- **Combined Synthesis**: LLM merges both perspectives
- **Query Modes**: Naive, Local, Global, Hybrid

## Directory Structure

```
KG_code/
├── KG_code_ALL/                    # Merged code KG (monolithic)
│   └── graph.json
├── KG_LaTeX/                       # LaTeX KG
│   └── graph.json
└── (Individual chapter KGs can be removed or ignored as they are no longer part of the streamlined pipeline)

lightrag_code_index/                 # Code KG LightRAG index
├── graph_chunk_entity_relation.graphml
├── kv_store_text_chunks.json
├── vdb_chunks.json
├── vdb_entities.json
└── vdb_relationships.json

lightrag_latex_index/                # LaTeX KG LightRAG index
└── (same structure as above)
```

## Model Selection Options

1. **Claude** → `anthropic/claude-sonnet-4.5`
2. **Gemini** → `google/gemini-2.5-flash`
3. **GPT-5** → `openai/gpt-5`
4. **Custom** → Enter any OpenRouter model ID

## Environment Setup

Create `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Start Ollama for embeddings:
```bash
ollama pull bge-m3:latest
ollama serve
```

## Key Features

✅ **Unified Code KG**: All thesis chapters merged into single queryable graph
✅ **Separate Theory/Implementation**: Clear distinction between LaTeX and Code
✅ **3-Response System**: Code answer + LaTeX answer + Combined synthesis
✅ **Meaningful Chunk IDs**: Traceable to source files
✅ **OpenRouter Integration**: Choose any LLM model
✅ **Local Embeddings**: No external embedding API needed (Ollama)

## Example Query Session

```
User: "How does the CNN model work?"

Response 1 (Code KG):
"The CNN is implemented in src/deep_learning/model.py with get_model() 
function. It uses a 5-layer encoder-decoder architecture with 
BatchNormalization and Dropout(0.5)..."

Response 2 (LaTeX KG):
"According to Chapter 2, the CNN uses convolution layers to extract 
spatial features from magnetic field maps. The loss function is 
L = Σ log p(B|geometry, θ)..."

Response 3 (Combined):
"The thesis describes a CNN for magnetic field prediction (Chapter 2). 
The theory explains convolution-based feature extraction, while the 
implementation in model.py shows a 5-layer network with specific 
architectural choices like BatchNorm and Dropout..."
```

## Requirements

- Python 3.8+
- OPENROUTER_API_KEY environment variable
- Local Ollama server running (`ollama serve`)
- Dependencies: `openai`, `lightrag`, `tiktoken`, `sentence-transformers`

## Troubleshooting

**Empty chunks?**
- Ensure `--source-dir` points to correct base directory
- Check that source files exist at paths in KG

**Integration fails?**
- Verify chunks exist in KG (run enhanced_chunks.py first)
- Check OPENROUTER_API_KEY is set
- Ensure Ollama is running (`ollama serve`)

**Query returns no results?**
- Verify both lightrag_code_index and lightrag_latex_index exist
- Check that KGs were integrated successfully
- Try different query modes (hybrid recommended)
