from tree_sitter import Language, Parser, QueryCursor
from pathlib import Path
import json

# Load the installed python grammar
from tree_sitter_python import language as python_language

PY_LANGUAGE = Language(python_language())
parser = Parser(PY_LANGUAGE)

# Simplified queries for Python
definitions_query = PY_LANGUAGE.query('''
(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function
''')

references_query = PY_LANGUAGE.query('''
(import_statement
  (dotted_name (identifier) @name.reference.import)) @reference.import

(import_from_statement
  module_name: (dotted_name (identifier) @name.reference.import_from)
  name: (dotted_name (identifier) @name.reference.import_from_name)) @reference.import_from

(call
  function: [
    (identifier) @name.reference.call
    (attribute
      attribute: (identifier) @name.reference.call)
  ]) @reference.call
''')

def get_calling_entity(node, file_stem):
    """
    Traverse up the AST to find the containing class and function.
    Returns a hierarchical name like 'file_stem::ClassName::function_name'.
    """
    current = node.parent
    containing_func = None
    containing_class = None

    while current:
        if current.type == 'function_definition' and not containing_func:
            name_node = current.child_by_field_name('name')
            if name_node:
                containing_func = name_node.text.decode('utf8')
        elif current.type == 'class_definition' and not containing_class:
            name_node = current.child_by_field_name('name')
            if name_node:
                containing_class = name_node.text.decode('utf8')
        
        current = current.parent

    parts = [file_stem]
    if containing_class:
        parts.append(containing_class)
    if containing_func:
        parts.append(containing_func)
    
    return "::".join(parts) if len(parts) > 1 else file_stem


def extract_symbols(code: str, file_path: str):
    """
    Extracts definitions and references from Python code using tree-sitter.
    """
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    file_stem = Path(file_path).stem

    definitions = []
    references = []

    # --- Extract Definitions ---
    cursor = QueryCursor(definitions_query)
    def_captures = cursor.captures(root_node)

    # Flatten captures into (node, capture_name) pairs
    def_capture_pairs = []
    for capture_name, nodes in def_captures.items():
        for node in nodes:
            def_capture_pairs.append((node, capture_name))

    for node, capture_name in def_capture_pairs:
        entity_name = node.text.decode('utf8')
        entity_type = capture_name.split('.')[-1] + 's' # e.g., 'classes', 'functions'
        
        # Build hierarchical name
        hierarchical_name = file_stem
        parent = node.parent
        is_method = False
        while parent:
            if parent.type == 'class_definition':
                is_method = True
                parent_name_node = parent.child_by_field_name('name')
                if parent_name_node:
                    hierarchical_name = f"{file_stem}::{parent_name_node.text.decode('utf8')}::{entity_name}"
                break
            parent = parent.parent
        
        if not is_method:
             hierarchical_name = f"{file_stem}::{entity_name}"


        definitions.append({
            "name": hierarchical_name,
            "type": entity_type,
            "category": "definition",
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "file_path": file_path,
            "code": node.text.decode('utf8')
        })

    # --- Extract References ---
    ref_cursor = QueryCursor(references_query)
    ref_captures = ref_cursor.captures(root_node)

    # Flatten captures into (node, capture_name) pairs
    ref_capture_pairs = []
    for capture_name, nodes in ref_captures.items():
        for node in nodes:
            ref_capture_pairs.append((node, capture_name))

    for node, capture_name in ref_capture_pairs:
        ref_name = node.text.decode('utf8')
        ref_type = capture_name.split('.')[-1]
        
        # Special handling for import_from which has two parts
        if ref_type == 'import_from_name':
            continue # We process this with import_from
        
        if ref_type == 'import_from':
            ref_type = 'imports'
            # Find the associated name part
            name_node = node.parent.child_by_field_name('name')
            if name_node:
                imported_names = [child.text.decode('utf8') for child in name_node.children if child.type == 'identifier']
                ref_name = f"{ref_name}.{'.'.join(imported_names)}"

        # Determine the calling entity
        calling_entity = get_calling_entity(node, file_stem)

        references.append({
            "name": ref_name,
            "type": ref_type + 's',
            "category": "reference",
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "file_path": file_path,
            "code": node.text.decode('utf8'),
            "calling_entity": calling_entity
        })

    return {
        "definitions": definitions,
        "references": references
    }

# Main execution block for testing
if __name__ == '__main__':
    import os
    # Example usage:
    # Create a dummy test file
    test_code = """
import os
from pathlib import Path

class MySimpleClass:
    def __init__(self):
        self.name = "MySimpleClass"

    def greet(self, message):
        print(f"Hello, {message} from {self.name}")
        
def standalone_function():
    p = Path(".")
    my_instance = MySimpleClass()
    my_instance.greet("World")
"""
    test_file_path = "test_script.py"
    with open(test_file_path, "w") as f:
        f.write(test_code)

    # Extract symbols from the dummy file
    extracted_data = extract_symbols(test_code, test_file_path)

    # Print the extracted data as JSON
    print(json.dumps(extracted_data, indent=2))

    # Clean up the dummy file
    os.remove(test_file_path)
