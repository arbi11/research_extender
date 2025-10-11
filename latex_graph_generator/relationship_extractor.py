import re
import sympy
from sympy.parsing.latex import parse_latex

def extract_mathematical_relationships(parsed_content):
    relationships = []

    equations = parsed_content.get('equations', [])
    sections = parsed_content.get('sections', [])

    # Build equation dependency graph through shared variables
    for i, eq in enumerate(equations):
        try:
            latex = eq.get('latex', '')
            if latex:
                expr = parse_latex(latex)
                variables = list(expr.free_symbols)

                # Find relationships with other equations
                for j, other_eq in enumerate(equations):
                    if i != j:
                        other_latex = other_eq.get('latex', '')
                        if other_latex:
                            other_expr = parse_latex(other_latex)
                            other_vars = list(other_expr.free_symbols)

                            # Shared variables create relationships
                            shared_vars = set(variables) & set(other_vars)
                            if shared_vars:
                                relationships.append({
                                    'type': 'variable_sharing',
                                    'equation_1': eq.get('id', i),
                                    'equation_2': other_eq.get('id', j),
                                    'shared_variables': list(shared_vars),
                                    'equation_1_latex': latex,
                                    'equation_2_latex': other_latex
                                })

        except Exception:
            continue

    # Extract theorem/proof relationships
    theorem_pattern = r'\\begin\{theorem\}(.*?)\\end\{theorem\}'
    proof_pattern = r'\\begin\{proof\}(.*?)\\end\{proof\}'
    definition_pattern = r'\\begin\{definition\}(.*?)\\end\{definition\}'

    for section in sections:
        content = section.get('content', '')
        theorems = re.findall(theorem_pattern, content, re.DOTALL)
        proofs = re.findall(proof_pattern, content, re.DOTALL)
        definitions = re.findall(definition_pattern, content, re.DOTALL)

        if theorems:
            relationships.append({
                'type': 'theorem_block',
                'section': section.get('title', ''),
                'theorem_count': len(theorems),
                'content_sample': theorems[0][:200] + '...' if theorems[0] else ''
            })

        if proofs:
            relationships.append({
                'type': 'proof_block',
                'section': section.get('title', ''),
                'proof_count': len(proofs),
                'content_sample': proofs[0][:200] + '...' if proofs[0] else ''
            })

        if definitions:
            relationships.append({
                'type': 'definition_block',
                'section': section.get('title', ''),
                'definition_count': len(definitions),
                'content_sample': definitions[0][:200] + '...' if definitions[0] else ''
            })

    # Extract citation relationships
    citation_pattern = r'\\cite\{([^}]+)\}'
    for section in sections:
        content = section.get('content', '')
        citations = re.findall(citation_pattern, content)
        if citations:
            relationships.append({
                'type': 'citations',
                'section': section.get('title', ''),
                'cited_works': [cite.strip() for cite in citations[0].split(',')],
                'citation_count': len(citations)
            })

    # Extract cross-references between equations and text
    ref_pattern = r'\\(?:eq)?ref\{([^}]+)\}'
    for section in sections:
        content = section.get('content', '')
        refs = re.findall(ref_pattern, content)
        if refs:
            relationships.append({
                'type': 'cross_reference',
                'section': section.get('title', ''),
                'referenced_items': refs,
                'reference_count': len(refs)
            })

    return relationships