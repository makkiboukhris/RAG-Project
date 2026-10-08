"""
Tree-sitter based code chunker.

Splits source files into self-contained chunks (one per top-level function /
class, with the file's imports and any types/globals it uses pulled in).

Entry points, from smallest to largest unit of work:

    chunk_source(source_bytes, ext)  -> list[str]      (no disk access)
    chunk_file(file_path)            -> list[str]
    chunk_folder(folder_path)        -> ChunkingResult (chunks + skipped files)

None of these print or exit - problems are logged and reported in the return
value, so they are safe to call from a web request.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field

from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

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


@dataclass
class Chunk:
    """One chunk of source text plus where it came from."""
    file: str         # path relative to the indexed folder, with "/" separators
    chunk_index: int  # position of this chunk within its file
    text: str

    def to_dict(self):
        return asdict(self)


@dataclass
class SkippedFile:
    """A file that was found but could not be chunked."""
    file: str
    reason: str


@dataclass
class ChunkingResult:
    chunks: list[Chunk] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)

    @property
    def file_count(self):
        return len({c.file for c in self.chunks})


# Folders that are never worth indexing. Hidden folders/files (names starting
# with ".") are always skipped as well.
DEFAULT_IGNORED_DIRS = frozenset({
    'node_modules', '__pycache__', 'venv', 'env', 'dist', 'build',
})


def supported_extensions():
    """Extensions that get real AST-based chunking (anything else that is
    valid UTF-8 text is kept as a single whole-file chunk)."""
    return sorted(LANGUAGE_MAP)


def chunk_source(source_bytes, ext):
    """Chunk one file's content. `ext` is the file extension including the
    dot (e.g. ".py") and selects the language. Accepts bytes or str.

    Raises UnicodeDecodeError if the content is not valid UTF-8.
    """
    if isinstance(source_bytes, str):
        source_bytes = source_bytes.encode('utf-8')

    language = LANGUAGE_MAP.get(ext.lower())
    if language is None:
        # Fallback for HTML, CSS, JSON, Markdown, etc.
        whole = source_bytes.decode('utf-8')
        return [whole] if whole.strip() else []

    # A Parser is cheap to create and is not safe to share between threads,
    # so each call gets its own.
    parser = Parser()
    parser.language = language
    tree = parser.parse(source_bytes)
    lines = source_bytes.split(b'\n')
    return extract_chunks_from_tree(source_bytes, lines, tree)


def chunk_file(file_path):
    """Chunk a single file on disk. Returns the list of chunk texts."""
    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    ext = os.path.splitext(file_path)[1]
    return chunk_source(source_bytes, ext)


def relative_file_path(folder_path, file_path):
    """`file_path` (absolute, or relative to `folder_path`) as it is stored in
    the index: relative to the folder, with "/" separators.

    Raises ValueError if the file is not inside the folder.
    """
    folder_path = os.path.abspath(os.fspath(folder_path))
    file_path = os.fspath(file_path)
    if not os.path.isabs(file_path):
        file_path = os.path.join(folder_path, file_path)
    try:
        rel_path = os.path.relpath(file_path, folder_path)
    except ValueError:  # different drive on Windows
        raise ValueError(f"{file_path} is not inside {folder_path}") from None
    rel_path = rel_path.replace(os.sep, '/')
    if rel_path == '..' or rel_path.startswith('../'):
        raise ValueError(f"{file_path} is not inside {folder_path}")
    return rel_path


def is_ignored_path(rel_path, ignored_dirs=DEFAULT_IGNORED_DIRS):
    """True if chunk_folder() would skip this path (a hidden or ignored folder
    anywhere above it, or a hidden file name)."""
    *dirs, name = rel_path.split('/')
    return name.startswith('.') or any(
        d.startswith('.') or d in ignored_dirs for d in dirs
    )


def chunk_one_file(folder_path, file_path):
    """Chunk a single file the way chunk_folder() would, returning Chunk
    objects tagged with its path relative to `folder_path`.

    Returns [] for a binary file. Raises whatever chunking raises (e.g.
    UnicodeDecodeError) and FileNotFoundError if the file doesn't exist.
    """
    rel_path = relative_file_path(folder_path, file_path)
    abs_path = os.path.join(os.path.abspath(os.fspath(folder_path)), rel_path)
    if not is_text_file(abs_path):
        return []
    return [Chunk(file=rel_path, chunk_index=i, text=text)
            for i, text in enumerate(chunk_file(abs_path))]


def chunk_folder(folder_path, ignored_dirs=DEFAULT_IGNORED_DIRS):
    """Walk `folder_path` and chunk every text file in it.

    Returns a ChunkingResult: `chunks` (each with the file it came from,
    relative to `folder_path`) and `skipped` (files that raised an error,
    with the reason). Binary files and hidden/ignored paths are left out
    silently.

    Raises NotADirectoryError if `folder_path` is not an existing folder.
    """
    folder_path = os.fspath(folder_path)
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    result = ChunkingResult()

    for root, dirs, files in os.walk(folder_path):
        # Prune in place so os.walk doesn't descend into these at all, and
        # sort so the chunk order is the same on every OS.
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith('.') and d not in ignored_dirs)

        for file in sorted(files):
            if file.startswith('.'):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, folder_path).replace(os.sep, '/')

            try:
                if not is_text_file(file_path):
                    continue
                for i, text in enumerate(chunk_file(file_path)):
                    result.chunks.append(Chunk(file=rel_path, chunk_index=i, text=text))
            except Exception as e:
                logger.warning("Skipping %s: %s", file_path, e)
                result.skipped.append(SkippedFile(file=rel_path, reason=str(e)))

    return result
