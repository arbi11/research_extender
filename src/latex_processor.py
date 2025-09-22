import re
from pathlib import Path
from plasTeX.TeX import TeX
from plasTeX.Renderers.Text import Renderer
from plasTeX import Document

import json


def process_latex(latex_file_path):
    """Process LaTeX file and extract structured content"""

    tex = TeX()
    tex.input(Path(latex_file_path).read_text(encoding='utf-8'))
    document = tex.parse()

    # Initialize data structure
    latex_data = {
        'title': '',
        'abstract': '',
        'sections': [],
        'equations': [],
        'citations': [],
        'figures': [],
        'algorithms': [],
        'bibliography': {}
    }

    # Extract title
    title_node = document.getElementsByTagName('title')
    if title_node:
        latex_data['title'] = str(title_node[0].textContent).strip()

    # Extract abstract
    abstract_node = document.getElementsByTagName('abstract')
    if abstract_node:
        latex_data['abstract'] = str(abstract_node[0].textContent).strip()

    # Extract sections with hierarchical structure
    section_tags = ['section', 'subsection', 'subsubsection', 'chapter', 'part']
    for tag in section_tags:
        sections = document.getElementsByTagName(tag)
        for section in sections:
            section_data = {
                'type': tag,
                'title': str(section.getAttribute('title') or section.textContent.split('\n')[0]).strip(),
                'content': str(section.textContent).strip(),
                'level': section_tags.index(tag) + 1,
                'equations_referenced': extract_equation_refs(str(section.textContent)),
                'citations_referenced': extract_citation_refs(str(section.textContent))
            }
            latex_data['sections'].append(section_data)

    # Extract equations
    equation_tags = ['equation', 'align', 'gather', 'multline', 'eqnarray']
    equation_counter = 1
    for tag in equation_tags:
        equations = document.getElementsByTagName(tag)
        for eq in equations:
            eq_data = {
                'id': f'eq_{equation_counter}',
                'type': tag,
                'content': eq.textContent.strip(),
                'label': eq.getAttribute('label') or f'eq_{equation_counter}',
                'raw_latex': eq.textContent.strip()
            }
            latex_data['equations'].append(eq_data)
            equation_counter += 1

    # Extract inline math as well
    inline_math = re.findall(r'\$([^$]+)\$', document.textContent)
    for i, math in enumerate(inline_math):
        eq_data = {
            'id': f'inline_eq_{i+1}',
            'type': 'inline',
            'content': math.strip(),
            'label': f'inline_eq_{i+1}',
            'raw_latex': f'${math}$'
        }
        latex_data['equations'].append(eq_data)

    # Extract citations
    cite_nodes = document.getElementsByTagName('cite')
    for cite in cite_nodes:
        citation_key = str(cite.textContent).strip()
        latex_data['citations'].append({
            'key': citation_key,
            'context': get_citation_context(document, cite)
        })

    # Extract figures
    figure_nodes = document.getElementsByTagName('figure')
    for fig in figure_nodes:
        caption_node = fig.getElementsByTagName('caption')
        caption = str(caption_node[0].textContent).strip() if caption_node else ''

        fig_data = {
            'label': str(fig.getAttribute('label')) or f'fig_{len(latex_data["figures"])+1}',
            'caption': caption,
            'content': str(fig.textContent).strip()
        }
        latex_data['figures'].append(fig_data)

    # Extract algorithms
    algorithm_nodes = document.getElementsByTagName('algorithm')
    for alg in algorithm_nodes:
        alg_data = {
            'label': str(alg.getAttribute('label')) or f'alg_{len(latex_data["algorithms"])+1}',
            'content': str(alg.textContent).strip(),
            'caption': ''
        }
        caption_node = alg.getElementsByTagName('caption')
        if caption_node:
            alg_data['caption'] = str(caption_node[0].textContent).strip()
        latex_data['algorithms'].append(alg_data)

    # Extract bibliography entries
    bibitem_nodes = document.getElementsByTagName('bibitem')
    for bib in bibitem_nodes:
        key = str(bib.getAttribute('key')) or str(bib.textContent).split()[0]
        latex_data['bibliography'][key] = str(bib.textContent).strip()

    return latex_data


def extract_equation_refs(text):
    """Extract equation references from text"""
    refs = re.findall(r'\\(?:eq)?ref\{([^}]+)\}', text)
    return list(set(refs))


def extract_citation_refs(text):
    """Extract citation references from text"""
    refs = re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}', text)
    citations = []
    for ref in refs:
        citations.extend(ref.split(','))
    return [c.strip() for c in citations]


def get_citation_context(document, cite_node):
    """Get surrounding context for a citation"""
    parent = cite_node.parentNode
    if parent:
        return str(parent.textContent).strip()[:200] + "..."
    return ""


def extract_mathematical_concepts(latex_data):
    """Extract and categorize mathematical concepts from equations"""

    math_concepts = {
        'variables': set(),
        'functions': set(),
        'operators': set(),
        'constants': [],
        'theorems': [],
        'definitions': []
    }

    # Pattern matching for mathematical elements
    variable_pattern = r'[a-zA-Z](?:_\{[^}]+\})?(?:\^\{[^}]+\})?'
    function_pattern = r'\\(?:sin|cos|tan|log|exp|sqrt|sum|int|prod|lim|max|min|arg(?:max|min))'
    operator_pattern = r'[+\-*/=<>≤≥∈∉⊂⊃∪∩]|\\(?:in|not|subset|cup|cap|le|ge|neq|pm|cdot|times)'

    for eq in latex_data['equations']:
        content = eq['content']

        # Extract variables
        variables = re.findall(variable_pattern, content)
        math_concepts['variables'].update(variables)

        # Extract functions
        functions = re.findall(function_pattern, content)
        math_concepts['functions'].update(functions)

        # Extract operators
        operators = re.findall(operator_pattern, content)
        math_concepts['operators'].update(operators)

    # Look for theorems and definitions in sections
    for section in latex_data['sections']:
        content = section['content'].lower()
        if 'theorem' in content or 'lemma' in content or 'proposition' in content:
            math_concepts['theorems'].append({
                'section': section['title'],
                'content': section['content'][:300] + "..."
            })

        if 'definition' in content or 'define' in content:
            math_concepts['definitions'].append({
                'section': section['title'],
                'content': section['content'][:300] + "..."
            })

    # Convert sets to lists for JSON serialization
    math_concepts['variables'] = list(math_concepts['variables'])
    math_concepts['functions'] = list(math_concepts['functions'])
    math_concepts['operators'] = list(math_concepts['operators'])

    return math_concepts


def main():
    """Example usage"""
    current_dir = Path(__file__).parent
    latex_file = current_dir.absolute() / "My_Thesis_3" / "My Thesis 2" / "Chapters" / "Chp1_Intro.tex" 

    # For demo, assuming we have extracted LaTeX from PDF
    print("Processing LaTeX file...")
    latex_data = process_latex(latex_file)

    print(f"Title: {latex_data['title']}")
    print(f"Sections found: {len(latex_data['sections'])}")
    print(f"Equations found: {len(latex_data['equations'])}")
    print(f"Citations found: {len(latex_data['citations'])}")

    # Extract mathematical concepts
    math_concepts = extract_mathematical_concepts(latex_data)
    print(f"Mathematical variables: {len(math_concepts['variables'])}")
    print(f"Mathematical functions: {len(math_concepts['functions'])}")

    # Save latex_data as JSON
    output_file = current_dir / "latex_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(latex_data, f, indent=2, ensure_ascii=False)
    print(f"Saved latex_data to: {output_file}")

    # Save math_concepts as JSON
    math_output_file = current_dir / "math_concepts.json"
    with open(math_output_file, 'w', encoding='utf-8') as f:
        json.dump(math_concepts, f, indent=2, ensure_ascii=False)
    print(f"Saved math_concepts to: {math_output_file}")

    return latex_data, math_concepts

if __name__ == "__main__":
    main()
