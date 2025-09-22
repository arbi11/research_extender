# Research Extension System

A clean, simple system that generates knowledge graphs from LaTeX papers and codebases, then uses LLM assistance to extend both code and paper with missing implementations and theoretical content.

## Core Components

### 1. `latex_processor.py`
Parses LaTeX files using plasTeX and extracts structured content:
- Sections, equations, citations, figures, algorithms
- Mathematical concept extraction
- Reference mapping

### 2. `knowledge_graph_builder.py`
Builds unified knowledge graphs using enhanced LightRAG:
- Code AST parsing for functions, classes, imports
- Equation-to-code mapping using pattern matching
- Gap detection between theory and implementation

### 3. `research_extender.py`
Generates extensions using LLM assistance:
- Identifies implementation gaps via RAG + LLM analysis
- Generates missing code implementations
- Creates paper extensions with proper LaTeX formatting

### 4. `run_extension_pipeline.py`
Complete pipeline runner with CLI interface.

## Installation

```bash
cd src/
uv sync  # or pip install -e .
```

## Usage

### Basic Usage
```bash
python run_extension_pipeline.py --latex paper.tex --code src/ --output ./extensions
```

### Advanced Usage
```bash
python run_extension_pipeline.py \
    --latex paper.tex \
    --code src1/ src2/ specific_file.py \
    --output ./my_extensions \
    --graph ./my_knowledge_graph
```

## Environment Setup

Create `.env` file with:
```
ANTHROPIC_API_KEY=your_claude_api_key_here
```

## Output Structure

```
extensions/
├── impl_equation_1.py          # Generated implementations
├── algorithm_deep_learning.py  # Generated algorithms
├── implementation_details.tex  # LaTeX extensions
├── extended_algorithms.tex
├── experimental_validation.tex
└── extension_summary_*.json    # Generation report
```

## Key Features

- **Clean Implementation**: No unnecessary try-except blocks or file checks
- **Minimal Dependencies**: Core focus on plasTeX, LightRAG, and Anthropic
- **Direct Processing**: Assumes valid inputs, no intermediate validation
- **Focused Scripts**: Each script has single, clear purpose
- **LLM-Enhanced**: Uses Claude for intelligent gap detection and code generation

## Requirements

- Python 3.8+
- ANTHROPIC_API_KEY environment variable
- Valid LaTeX source files (.tex format)
- Code in supported languages (Python, C++, JavaScript, Java)