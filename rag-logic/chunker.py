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

# 3. Define AST node types that represent "chunks" across different languages
TARGET_NODE_TYPES = {
    'function_definition', 'class_definition', 'method_definition', 
    'function_declaration', 'class_declaration', 'method_declaration',
    'struct_specifier', 'interface_declaration',
    'lexical_declaration',   # Catches 'const' and 'let' (used for arrow functions)
    'variable_declaration'   # Catches 'var'
}

def is_text_file(filepath):
    try:
        with open(filepath, 'tr', encoding='utf-8') as f:
            f.read(1024)
            return True
    except UnicodeDecodeError:
        return False

def get_preceding_comments(source_bytes, start_byte, lines):
    """Traces backwards in the raw text to capture comments right above a node."""
    # Find which line the start_byte corresponds to
    byte_count = 0
    start_line_idx = 0
    for i, line in enumerate(lines):
        # +1 for newline character
        if byte_count + len(line) + 1 > start_byte:
            start_line_idx = i
            break
        byte_count += len(line) + 1

    # Scan backwards for comment lines
    current_idx = start_line_idx - 1
    while current_idx >= 0:
        stripped = lines[current_idx].strip().decode('utf-8')
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            current_idx -= 1
        else:
            break
            
    # Return the byte offset of the earliest comment line found
    if current_idx == start_line_idx - 1:
        return start_byte
        
    earliest_line = current_idx + 1
    return sum(len(line) + 1 for line in lines[:earliest_line])

def extract_and_print_chunks(folder_path):
    chunks = []
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
                    # Parse with Tree-Sitter
                    parser.language = LANGUAGE_MAP[ext]
                    tree = parser.parse(source_bytes)
                    
                    # Walk the AST to find top-level functions and classes
                    cursor = tree.walk()
                    for node in tree.root_node.children:
                        if node.type in TARGET_NODE_TYPES:
                            # Capture preceding comments by tracing bytes
                            start_byte = get_preceding_comments(source_bytes, node.start_byte, lines)
                            end_byte = node.end_byte
                            
                            chunk = source_bytes[start_byte:end_byte].decode('utf-8')
                            chunks.append(chunk)
                else:
                    # Fallback for HTML, CSS, JSON, Markdown, etc.
                    chunks.append(source_bytes.decode('utf-8'))
                    
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*20} CHUNK {i} {'='*20}\n")
        print(chunk)
        
    return chunks

if __name__ == "__main__":
    extract_and_print_chunks("test-files")