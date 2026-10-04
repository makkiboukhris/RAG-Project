import os
import tree_sitter
from tree_sitter import Language, Parser

# 1. Import language binaries
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_cpp
import tree_sitter_go

# 2. Map extensions to their Tree-Sitter language objects
LANGUAGE_MAP = {
    '.py': Language(tree_sitter_python.language()),
    '.js': Language(tree_sitter_javascript.language()),
    '.jsx': Language(tree_sitter_javascript.language()),
    '.ts': Language(tree_sitter_typescript.language_typescript()),
    '.tsx': Language(tree_sitter_typescript.language_tsx()),
    '.java': Language(tree_sitter_java.language()),
    '.cpp': Language(tree_sitter_cpp.language()),
    '.h': Language(tree_sitter_cpp.language()),
    '.go': Language(tree_sitter_go.language()),
}

# 3. AST node types that represent a "chunkable" function/method unit
#    (but NOT classes - classes get their own handling below, since a class
#    is chunked as a single whole-class unit rather than split per-method).
FUNCTION_NODE_TYPES = {
    'function_definition',    # Python, C/C++
    'function_declaration',   # JS/TS, Go, Java (Go methods-with-receiver are
                               # still `function_declaration` in tree-sitter-go)
    'method_definition',      # JS/TS class methods (only reached if a class
                               # method somehow appears at top level)
    'method_declaration',     # Java
}

# A class/struct-like container that should be chunked as ONE unit (itself
# plus all its methods together), and whose name is also registered as a
# type so other chunks that reference it get it embedded too.
CLASS_NODE_TYPES = {
    'class_definition',   # Python
    'class_declaration',  # JS/TS, Java
}

# Node types that define a *name* which can be referenced elsewhere in the
# file as a type (interfaces, structs, enums, type aliases...). When a
# chunk uses one of these names, the definition is pulled into the chunk so
# the chunk is self-contained.
TYPE_DEF_NODE_TYPES = {
    'interface_declaration',   # TS
    'type_alias_declaration',  # TS
    'enum_declaration',        # TS/Java
    'struct_specifier',        # C/C++
    'class_specifier',         # C++
    'union_specifier',         # C/C++
    'enum_specifier',          # C/C++
    'type_declaration',        # Go (`type Foo struct {...}` / `type Foo int`)
}

# File-header nodes (imports, package clauses, includes) that should be
# repeated verbatim at the top of every chunk extracted from the file, so
# each chunk is independently valid/parseable.
HEADER_NODE_TYPES = {
    'import_statement', 'import_from_statement',  # Python, JS/TS
    'import_declaration',                          # Java
    'preproc_include',                             # C/C++
    'package_clause',                               # Go
}

# Top-level "container" nodes that hold global variable/constant
# declarations in JS/TS and Go. (Python and C/C++ globals are handled via
# their own node shapes - see extract_names/the main loop below.)
GLOBAL_VAR_NODE_TYPES = {
    'lexical_declaration', 'variable_declaration',  # JS/TS (let/const/var)
    'var_declaration', 'const_declaration',         # Go
}

# Node types whose presence inside a declaration's value means "this isn't
# really a plain global variable, it's a function" (e.g. `const f = () => {}`)
FUNCTION_VALUE_NODE_TYPES = {
    'arrow_function', 'function', 'function_expression', 'func_literal',
}

# Some languages wrap a real top-level declaration in an outer syntactic
# node: JS/TS `export ...` statements, and Python `@decorator` definitions.
# Maps the wrapper node type -> the field name that holds the real node.
WRAPPER_FIELD_BY_TYPE = {
    'export_statement': 'declaration',
    'export_default_declaration': 'declaration',
    'decorated_definition': 'definition',  # Python decorators
}


def is_text_file(filepath):
    try:
        with open(filepath, 'tr', encoding='utf-8') as f:
            f.read(1024)
            return True
    except UnicodeDecodeError:
        return False


def node_text(source_bytes, node):
    return source_bytes[node.start_byte:node.end_byte].decode('utf-8')


def get_preceding_comments(source_bytes, start_byte, lines):
    """Traces backwards in the raw text to capture comments right above a node."""
    byte_count = 0
    start_line_idx = 0
    for i, line in enumerate(lines):
        if byte_count + len(line) + 1 > start_byte:
            start_line_idx = i
            break
        byte_count += len(line) + 1

    current_idx = start_line_idx - 1
    while current_idx >= 0:
        stripped = lines[current_idx].strip().decode('utf-8')
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            current_idx -= 1
        else:
            break

    if current_idx == start_line_idx - 1:
        return start_byte

    earliest_line = current_idx + 1
    return sum(len(line) + 1 for line in lines[:earliest_line])


def text_with_comments(source_bytes, lines, node):
    start = get_preceding_comments(source_bytes, node.start_byte, lines)
    return source_bytes[start:node.end_byte].decode('utf-8')


def unwrap_wrapper(node):
    """Unwrap `export ...` (JS/TS) and `@decorator` (Python) wrappers.
    Returns (classification_node, span_node): classification_node is used to
    decide what kind of thing this is and to extract its name; span_node is
    used for the actual chunk/definition text so the wrapper syntax
    (export keyword / decorators) is preserved in the output."""
    field = WRAPPER_FIELD_BY_TYPE.get(node.type)
    if field:
        inner = node.child_by_field_name(field)
        if inner is not None:
            return inner, node
    return node, node


def collect_identifiers(node):
    """Collect every leaf identifier-like token used anywhere inside `node`.
    This is a heuristic (name-based, not scope-aware) way to tell whether a
    chunk "uses" a given global/type: it's deliberately a little
    over-inclusive (e.g. it doesn't account for shadowing) rather than risk
    leaving a chunk that references something undefined.
    """
    identifiers = set()
    stack = [node]
    while stack:
        current = stack.pop()
        children = current.children
        if not children:
            if current.type.endswith('identifier'):
                identifiers.add(current.text.decode('utf-8'))
        else:
            stack.extend(children)
    return identifiers


def contains_function_value(node):
    """True if `node` (e.g. a `const x = ...` declaration) has a function
    literal somewhere inside it, meaning it should be treated as its own
    chunkable unit rather than as a plain global variable."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in FUNCTION_VALUE_NODE_TYPES:
            return True
        stack.extend(current.children)
    return False


def unwrap_type_from_declaration(node):
    """C/C++: a top-level `struct Foo { ... };` / `class Foo { ... };` is
    parsed as a `declaration` node wrapping a `struct_specifier` /
    `class_specifier` / etc. Pull the inner specifier out so it can be
    treated as a type definition. Returns None if `node` isn't that shape,
    or if the specifier has no body (i.e. it's just a variable of that type,
    or a forward declaration, not a type definition)."""
    if node.type != 'declaration':
        return None
    for child in node.children:
        if child.type in ('struct_specifier', 'class_specifier', 'union_specifier', 'enum_specifier'):
            if child.child_by_field_name('body') is not None:
                return child
    return None


def find_first_identifier(node):
    stack = [node]
    while stack:
        current = stack.pop(0)
        if current.type == 'identifier':
            return current.text.decode('utf-8')
        stack = list(current.children) + stack
    return None


def extract_names(node):
    """Return the list of names a top-level node introduces (a type name,
    or one/more variable names for a declaration with multiple declarators)."""
    t = node.type

    if t in ('interface_declaration', 'type_alias_declaration', 'enum_declaration',
              'class_declaration', 'class_definition', 'struct_specifier',
              'class_specifier', 'union_specifier', 'enum_specifier'):
        name_node = node.child_by_field_name('name')
        return [name_node.text.decode('utf-8')] if name_node else []

    if t == 'type_declaration':  # Go
        names = []
        for child in node.children:
            if child.type == 'type_spec':
                n = child.child_by_field_name('name')
                if n:
                    names.append(n.text.decode('utf-8'))
        return names

    if t in ('lexical_declaration', 'variable_declaration'):  # JS/TS
        names = []
        for child in node.children:
            if child.type == 'variable_declarator':
                n = child.child_by_field_name('name')
                if n:
                    names.append(n.text.decode('utf-8'))
        return names

    if t in ('var_declaration', 'const_declaration'):  # Go
        names = []
        for child in node.children:
            if child.type in ('var_spec', 'const_spec'):
                n = child.child_by_field_name('name')
                if n:
                    names.append(n.text.decode('utf-8'))
                else:
                    for gc in child.children:
                        if gc.type == 'identifier':
                            names.append(gc.text.decode('utf-8'))
        return names

    if t == 'declaration':  # C/C++ plain global variable, e.g. `int counter = 0;`
        names = []
        for child in node.children:
            if child.type in ('init_declarator',):
                d = child.child_by_field_name('declarator')
                n = find_first_identifier(d) if d is not None else None
                if n:
                    names.append(n)
            elif child.type == 'identifier':
                names.append(child.text.decode('utf-8'))
        return names

    if t == 'expression_statement':  # Python top-level assignment, e.g. `X = 5`
        names = []
        for child in node.children:
            if child.type == 'assignment':
                left = child.child_by_field_name('left')
                if left is not None and left.type == 'identifier':
                    names.append(left.text.decode('utf-8'))
        return names

    return []


def extract_chunks_from_tree(source_bytes, lines, tree):
    root = tree.root_node

    header_parts = []   # imports / package clauses, in source order
    type_defs = {}       # name -> definition text (comments kept)
    global_defs = {}      # name -> definition text (NO leading comments -
                           # a comment above a bare global is treated as
                           # incidental, not documentation of a named type)
    function_nodes = []  # top-level nodes we'll split into their own chunks

    for raw_node in root.children:
        classification_node, span_node = unwrap_wrapper(raw_node)
        t = classification_node.type

        if t in HEADER_NODE_TYPES:
            header_parts.append(node_text(source_bytes, span_node))
            continue

        if t in CLASS_NODE_TYPES:
            # The whole class (itself + all its methods) is one chunk, and
            # its name is also registered as a type so any other chunk that
            # references it (e.g. as a parameter/return type) pulls it in.
            text = text_with_comments(source_bytes, lines, span_node)
            function_nodes.append(span_node)
            for name in extract_names(classification_node):
                type_defs[name] = text
            continue

        if t in FUNCTION_NODE_TYPES:
            function_nodes.append(span_node)
            continue

        inner_type = unwrap_type_from_declaration(classification_node)
        if inner_type is not None:
            text = text_with_comments(source_bytes, lines, span_node)
            for name in extract_names(inner_type):
                type_defs[name] = text
            continue

        if t in GLOBAL_VAR_NODE_TYPES:
            if contains_function_value(classification_node):
                # e.g. `const handler = () => {...}` behaves like a function
                function_nodes.append(span_node)
            else:
                text = node_text(source_bytes, span_node)
                for name in extract_names(classification_node):
                    global_defs[name] = text
            continue

        if t in TYPE_DEF_NODE_TYPES:
            text = text_with_comments(source_bytes, lines, span_node)
            for name in extract_names(classification_node):
                type_defs[name] = text
            continue

        # Plain C/C++ global variable (`int counter = 0;`) or Python
        # top-level assignment (`X = 5`) that isn't a type definition.
        if t in ('declaration', 'expression_statement'):
            if not contains_function_value(classification_node):
                names = extract_names(classification_node)
                if names:
                    text = node_text(source_bytes, span_node)
                    for name in names:
                        global_defs[name] = text
            continue

    header_text = "\n".join(header_parts)

    chunks = []
    for node in function_nodes:
        own_text = text_with_comments(source_bytes, lines, node)
        used_identifiers = collect_identifiers(node)

        dependency_parts = []
        for name, text in type_defs.items():
            if name in used_identifiers and text != own_text and text not in dependency_parts:
                dependency_parts.append(text)
        for name, text in global_defs.items():
            if name in used_identifiers and text != own_text and text not in dependency_parts:
                dependency_parts.append(text)

        pieces = [p for p in ([header_text] + dependency_parts + [own_text]) if p]
        chunks.append("\n\n".join(pieces))

    # Fallback: if nothing was recognized as a separately-chunkable
    # function/class (e.g. a C++ header with only a struct + a class whose
    # methods have no bodies), keep the whole file as a single chunk rather
    # than silently dropping its content.
    if not chunks:
        whole = source_bytes.decode('utf-8', errors='replace')
        if whole.strip():
            chunks.append(whole)

    return chunks


def extract_chunks_with_metadata(folder_path):
    """Like extract_and_print_chunks, but returns each chunk alongside the
    file it came from and its position within that file - metadata that's
    useful to keep around once you start embedding/indexing the chunks."""
    all_chunks = []
    parser = Parser()

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            if '/.' in file_path or '\\.' in file_path or not is_text_file(file_path):
                continue

            ext = os.path.splitext(file)[1].lower()

            try:
                with open(file_path, 'rb') as f:
                    source_bytes = f.read()
                lines = source_bytes.split(b'\n')

                if ext in LANGUAGE_MAP:
                    parser.language = LANGUAGE_MAP[ext]
                    tree = parser.parse(source_bytes)
                    file_chunks = extract_chunks_from_tree(source_bytes, lines, tree)
                else:
                    # Fallback for HTML, CSS, JSON, Markdown, etc.
                    file_chunks = [source_bytes.decode('utf-8')]

                for i, text in enumerate(file_chunks):
                    all_chunks.append({
                        "file": file_path,
                        "chunk_index": i,
                        "text": text,
                    })

            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    return all_chunks


def extract_and_print_chunks(folder_path):
    chunks = extract_chunks_with_metadata(folder_path)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*20} CHUNK {i} ({chunk['file']}) {'='*20}\n")
        print(chunk["text"])

    return [c["text"] for c in chunks]


if __name__ == "__main__":
    extract_and_print_chunks("test-files")