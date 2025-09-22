# Research Knowledge Graph Pipeline

This pipeline extracts LaTeX content, builds a knowledge graph using LightRAG, and generates research extensions using OpenRouter LLMs.

## Key Features

- **LaTeX Processing**: Parses LaTeX files to extract sections, equations, citations, and algorithms
- **Code Analysis**: Analyzes Python code to extract functions, classes, and imports
- **Knowledge Graph**: Uses LightRAG to build a unified knowledge base from LaTeX and code
- **Gap Detection**: Identifies missing implementations and theoretical gaps
- **Extension Generation**: Generates Python code and LaTeX sections for missing components
- **OpenRouter Integration**: Uses OpenRouter for LLM calls (supports Claude, GPT-4, etc.)
- **Fast Embeddings**: Uses sentence-transformers/all-MiniLM-L6-v2 for efficient embeddings

## Setup

1. Install dependencies:
```bash
pip install openai sentence-transformers lightrag plasTeX python-dotenv
```

2. Set up environment variables in `.env`:
```bash
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional
```

## Usage

### Basic Usage

```python
from knowledge_graph_builder import KnowledgeGraphBuilder
from research_extender import ResearchExtender
from latex_processor import process_latex

# Process LaTeX file
latex_data = process_latex("paper.tex")

# Build knowledge graph
builder = KnowledgeGraphBuilder(
    working_dir="./research_graph",
    llm_model="anthropic/claude-3.7-sonnet",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384
)

# Create research extender
extender = ResearchExtender(builder, llm_model="anthropic/claude-3.7-sonnet")

# Generate extensions
result = await extender.generate_extensions(
    latex_file_path="paper.tex",
    code_paths=["src/", "algorithms/"],
    output_dir="./extensions"
)
```

### Command Line Usage

```bash
python run_extension_pipeline.py --latex paper.tex --code src/ algorithms/ --output ./extensions
```

## Architecture

### KnowledgeGraphBuilder
- Parses code files and extracts structure
- Maps code implementations to mathematical equations
- Builds LightRAG knowledge base with embeddings
- Uses OpenRouter for LLM calls

### ResearchExtender
- Finds gaps between theory and implementation
- Generates Python code for missing implementations
- Creates LaTeX sections for new algorithms
- Produces comprehensive extension reports

### LaTeX Processor
- Extracts sections, equations, citations, figures
- Handles mathematical content and references
- Supports hierarchical document structure

## Configuration

### LLM Models (OpenRouter)
- `anthropic/claude-3.7-sonnet` (default)
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- Any OpenRouter-supported model

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast)
- `jinaai/jina-embeddings-v3` (1024 dim, better quality)

## Output Structure

```
extensions/
├── impl_equation_1.py          # Generated implementations
├── algorithm_optimization.py   # Generated algorithms
├── implementation_details.tex  # LaTeX sections
├── extended_algorithms.tex
├── experimental_validation.tex
└── extension_summary_*.json    # Summary report
```

## Example Output

The pipeline generates:
1. **Python implementations** for mathematical equations
2. **Algorithm classes** for theoretical concepts
3. **LaTeX sections** with proper academic formatting
4. **Summary reports** with gap analysis and metrics

## Dependencies

- `openai`: OpenRouter API client
- `sentence-transformers`: Fast embeddings
- `lightrag`: Knowledge graph and RAG
- `plasTeX`: LaTeX parsing
- `python-dotenv`: Environment variables
- `numpy`: Numerical operations
- `pathlib`: File handling
