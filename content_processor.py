#!/usr/bin/env python3
"""
Content Processor Module

This module handles the processing and chunking of repository content for embedding.
It provides functionality to:
1. Classify repository files by type
2. Process different file types (code, documentation, configuration)
3. Extract logical chunks with appropriate metadata
4. Prepare content for embedding

It implements a hierarchical logging system for debugging and monitoring the
content processing pipeline.
"""

import os
import re
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Generator
import ast
import tokenize
from io import StringIO
from collections import defaultdict

# Setup module logger
logger = logging.getLogger("content_processor")
logger.propagate = False
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Define file type constants
FILE_TYPE_CODE = "code"
FILE_TYPE_DOCUMENTATION = "documentation"
FILE_TYPE_CONFIGURATION = "configuration"
FILE_TYPE_UNKNOWN = "unknown"

# Define code language constants
LANG_PYTHON = "python"
LANG_JAVASCRIPT = "javascript"
LANG_TYPESCRIPT = "typescript"
LANG_JAVA = "java"
LANG_GO = "go"
LANG_RUBY = "ruby"
LANG_RUST = "rust"
LANG_CPP = "cpp"
LANG_CSHARP = "csharp"
LANG_UNKNOWN = "unknown"


class ContentChunk:
    """Represents a chunk of content with metadata for embedding."""

    def __init__(
        self,
        content: str,
        file_path: str,
        chunk_type: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        language: Optional[str] = None,
        parent: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a content chunk.

        Args:
            content: The text content of the chunk
            file_path: Path to the source file, relative to repository root
            chunk_type: Type of content (code, documentation, configuration)
            start_line: Starting line number in the source file (optional)
            end_line: Ending line number in the source file (optional)
            language: Programming language or format (optional)
            parent: Name of parent entity (e.g., class name for a method) (optional)
            name: Name of the chunk entity (e.g., function name) (optional)
            metadata: Additional metadata as key-value pairs (optional)
        """
        self.content = content
        self.file_path = file_path
        self.chunk_type = chunk_type
        self.start_line = start_line
        self.end_line = end_line
        self.language = language
        self.parent = parent
        self.name = name
        self.metadata = metadata or {}

        # Calculate token count (simple approximation)
        self.token_count = len(content.split())

    def __repr__(self) -> str:
        """String representation of the chunk."""
        return (f"ContentChunk(file='{self.file_path}', "
                f"type='{self.chunk_type}', "
                f"lines={self.start_line}-{self.end_line}, "
                f"name='{self.name}', "
                f"tokens={self.token_count})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "parent": self.parent,
            "name": self.name,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


class ContentProcessor:
    """
    Processes repository content for embedding.
    Handles file classification, content extraction, and chunking.
    """

    def __init__(self, repo_path: str, log_level: int = logging.INFO):
        """
        Initialize the content processor.

        Args:
            repo_path: Path to the repository root
            log_level: Logging level for this processor instance
        """
        self.repo_path = repo_path
        self.chunks = []

        # Setup processor-specific logger
        self.logger = logging.getLogger(f"content_processor.{os.path.basename(repo_path)}")
        self.logger.setLevel(log_level)

        # Create file handler for this repository
        log_dir = os.path.join(os.path.dirname(repo_path), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{os.path.basename(repo_path)}_processing.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Initialized content processor for {repo_path}")

        # Stats tracking
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "files_by_type": defaultdict(int),
            "chunks_by_type": defaultdict(int),
            "processing_time": 0,
            "errors": 0
        }

    def classify_file(self, file_path: str) -> Tuple[str, Optional[str]]:
        """
        Classify a file based on its extension and content.

        Args:
            file_path: Path to the file, relative to repository root

        Returns:
            Tuple of (file_type, language)
        """
        start_time = time.time()
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Initialize with unknown types
        file_type = FILE_TYPE_UNKNOWN
        language = LANG_UNKNOWN

        # Check documentation files
        if ext in ['.md', '.rst', '.txt', '.docx', '.pdf']:
            file_type = FILE_TYPE_DOCUMENTATION
            language = ext[1:]  # Use extension without the dot

        # Check configuration files
        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']:
            file_type = FILE_TYPE_CONFIGURATION
            language = ext[1:]  # Use extension without the dot

        # Check various code files
        elif ext in ['.py']:
            file_type = FILE_TYPE_CODE
            language = LANG_PYTHON
        elif ext in ['.js']:
            file_type = FILE_TYPE_CODE
            language = LANG_JAVASCRIPT
        elif ext in ['.ts', '.tsx']:
            file_type = FILE_TYPE_CODE
            language = LANG_TYPESCRIPT
        elif ext in ['.java']:
            file_type = FILE_TYPE_CODE
            language = LANG_JAVA
        elif ext in ['.go']:
            file_type = FILE_TYPE_CODE
            language = LANG_GO
        elif ext in ['.rb']:
            file_type = FILE_TYPE_CODE
            language = LANG_RUBY
        elif ext in ['.rs']:
            file_type = FILE_TYPE_CODE
            language = LANG_RUST
        elif ext in ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']:
            file_type = FILE_TYPE_CODE
            language = LANG_CPP
        elif ext in ['.cs']:
            file_type = FILE_TYPE_CODE
            language = LANG_CSHARP

        # Special cases by filename
        filename = os.path.basename(file_path)
        if filename in ['Dockerfile']:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'dockerfile'
        elif filename in ['.gitignore', '.dockerignore']:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'ignore'
        elif filename in ['Makefile', 'makefile']:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'makefile'

        # Special case for GitHub workflow files
        if '.github/workflows' in file_path and ext in ['.yml', '.yaml']:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'github_workflow'

        # Special case for package definition files
        if filename in ['package.json', 'package-lock.json', 'yarn.lock']:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'npm'
        elif filename in ['requirements.txt', 'Pipfile', 'Pipfile.lock', 'pyproject.toml', 'setup.py']:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'python_package'

        # Special case for GitHub-specific files
        if '.github/' in file_path:
            file_type = FILE_TYPE_CONFIGURATION
            language = 'github'

        elapsed = time.time() - start_time
        self.logger.debug(f"Classified {file_path} as {file_type}/{language} in {elapsed:.4f}s")

        return file_type, language

    def process_file(self, file_path: str) -> List[ContentChunk]:
        """
        Process a file and create content chunks.

        Args:
            file_path: Path to the file, relative to repository root

        Returns:
            List of ContentChunk objects
        """
        start_time = time.time()
        abs_path = os.path.join(self.repo_path, file_path)

        # Skip if file doesn't exist
        if not os.path.exists(abs_path):
            self.logger.warning(f"File does not exist: {abs_path}")
            return []

        # Skip directories
        if os.path.isdir(abs_path):
            self.logger.debug(f"Skipping directory: {abs_path}")
            return []

        try:
            # Read file content
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Classify file
            file_type, language = self.classify_file(file_path)

            # Process based on file type
            if file_type == FILE_TYPE_CODE:
                chunks = self._process_code_file(file_path, content, language)
            elif file_type == FILE_TYPE_DOCUMENTATION:
                chunks = self._process_documentation_file(file_path, content, language)
            elif file_type == FILE_TYPE_CONFIGURATION:
                chunks = self._process_configuration_file(file_path, content, language)
            else:
                # For unknown types, create a single chunk
                self.logger.info(f"Processing unknown file type: {file_path}")
                chunks = [ContentChunk(
                    content=content,
                    file_path=file_path,
                    chunk_type=FILE_TYPE_UNKNOWN,
                    language=language
                )]

            # Update stats
            self.stats["files_processed"] += 1
            self.stats["files_by_type"][file_type] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["chunks_by_type"][file_type] += len(chunks)

            elapsed = time.time() - start_time
            self.logger.info(
                f"Processed {file_path} ({file_type}/{language}): "
                f"created {len(chunks)} chunks in {elapsed:.4f}s"
            )

            return chunks

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            self.stats["errors"] += 1
            return []

    def _process_code_file(self, file_path: str, content: str, language: str) -> List[ContentChunk]:
        """
        Process a code file and create code chunks.

        Args:
            file_path: Path to the file
            content: File content
            language: Programming language

        Returns:
            List of ContentChunk objects
        """
        chunks = []

        # Use language-specific processing
        if language == LANG_PYTHON:
            chunks = self._process_python_code(file_path, content)
        else:
            # For other languages, use generic chunking
            self.logger.debug(f"Using generic chunking for {language} file: {file_path}")
            chunks = self._chunk_by_size(
                content, file_path, FILE_TYPE_CODE, language,
                chunk_size=1500, overlap=200
            )

        return chunks

    def _process_python_code(self, file_path: str, content: str) -> List[ContentChunk]:
        """
        Process Python code file using AST.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            List of ContentChunk objects
        """
        chunks = []

        try:
            # Parse the Python code into an AST
            tree = ast.parse(content)

            # Track line numbers for AST nodes
            line_numbers = {}
            for node in ast.walk(tree):
                if hasattr(node, 'lineno'):
                    line_numbers[node] = (
                        getattr(node, 'lineno', None),
                        getattr(node, 'end_lineno', None)
                    )

            # Process classes
            for cls_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                cls_name = cls_node.name
                cls_start, cls_end = line_numbers.get(cls_node, (None, None))
                if cls_start and cls_end:
                    # Extract class definition and docstring
                    cls_lines = content.splitlines()[cls_start-1:cls_end]
                    cls_content = '\n'.join(cls_lines)

                    # Create chunk for the class
                    cls_chunk = ContentChunk(
                        content=cls_content,
                        file_path=file_path,
                        chunk_type=FILE_TYPE_CODE,
                        start_line=cls_start,
                        end_line=cls_end,
                        language=LANG_PYTHON,
                        name=cls_name,
                        metadata={"type": "class"}
                    )
                    chunks.append(cls_chunk)

                    # Process methods within the class
                    for method_node in [n for n in cls_node.body if isinstance(n, ast.FunctionDef)]:
                        method_name = method_node.name
                        method_start, method_end = line_numbers.get(method_node, (None, None))
                        if method_start and method_end:
                            # Extract method definition
                            method_lines = content.splitlines()[method_start-1:method_end]
                            method_content = '\n'.join(method_lines)

                            # Create chunk for the method
                            method_chunk = ContentChunk(
                                content=method_content,
                                file_path=file_path,
                                chunk_type=FILE_TYPE_CODE,
                                start_line=method_start,
                                end_line=method_end,
                                language=LANG_PYTHON,
                                parent=cls_name,
                                name=method_name,
                                metadata={"type": "method"}
                            )
                            chunks.append(method_chunk)

            # Process standalone functions
            for func_node in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
                func_name = func_node.name
                func_start, func_end = line_numbers.get(func_node, (None, None))
                if func_start and func_end:
                    # Extract function definition
                    func_lines = content.splitlines()[func_start-1:func_end]
                    func_content = '\n'.join(func_lines)

                    # Create chunk for the function
                    func_chunk = ContentChunk(
                        content=func_content,
                        file_path=file_path,
                        chunk_type=FILE_TYPE_CODE,
                        start_line=func_start,
                        end_line=func_end,
                        language=LANG_PYTHON,
                        name=func_name,
                        metadata={"type": "function"}
                    )
                    chunks.append(func_chunk)

            # If no chunks were created (e.g., file with only imports or constants),
            # create a single chunk for the entire file
            if not chunks:
                self.logger.debug(f"No classes or functions found in {file_path}, using whole file")
                chunks = [ContentChunk(
                    content=content,
                    file_path=file_path,
                    chunk_type=FILE_TYPE_CODE,
                    language=LANG_PYTHON,
                    metadata={"type": "whole_file"}
                )]

            return chunks

        except SyntaxError as e:
            # Handle Python syntax errors
            self.logger.warning(f"Syntax error in Python file {file_path}: {e}")
            # Fall back to generic chunking
            return self._chunk_by_size(
                content, file_path, FILE_TYPE_CODE, LANG_PYTHON,
                chunk_size=1500, overlap=200
            )

        except Exception as e:
            self.logger.error(f"Error processing Python file {file_path}: {e}", exc_info=True)
            # Fall back to generic chunking
            return self._chunk_by_size(
                content, file_path, FILE_TYPE_CODE, LANG_PYTHON,
                chunk_size=1500, overlap=200
            )

    def _process_documentation_file(self, file_path: str, content: str, language: str) -> List[ContentChunk]:
        """
        Process a documentation file.

        Args:
            file_path: Path to the file
            content: File content
            language: Documentation format (e.g., md, rst)

        Returns:
            List of ContentChunk objects
        """
        chunks = []

        if language == 'md':
            # For Markdown, try to chunk by headings
            chunks = self._chunk_markdown_by_heading(file_path, content)
        else:
            # For other documentation formats, use generic chunking
            chunks = self._chunk_by_size(
                content, file_path, FILE_TYPE_DOCUMENTATION, language,
                chunk_size=1500, overlap=250
            )

        return chunks

    def _chunk_markdown_by_heading(self, file_path: str, content: str) -> List[ContentChunk]:
        """
        Chunk Markdown content by headings.

        Args:
            file_path: Path to the file
            content: Markdown content

        Returns:
            List of ContentChunk objects
        """
        chunks = []

        # Find all headings and their positions
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(content))

        # If no headings found, use generic chunking
        if not headings:
            self.logger.debug(f"No headings found in Markdown file {file_path}, using size-based chunking")
            return self._chunk_by_size(
                content, file_path, FILE_TYPE_DOCUMENTATION, 'md',
                chunk_size=1500, overlap=250
            )

        # Process content between headings
        for i, match in enumerate(headings):
            heading = match.group(0)
            heading_level = len(match.group(1))  # Number of # characters
            heading_text = match.group(2).strip()
            start_pos = match.start()

            # Determine end position (next heading or end of file)
            if i < len(headings) - 1:
                end_pos = headings[i + 1].start()
            else:
                end_pos = len(content)

            # Extract section content
            section_content = content[start_pos:end_pos]

            # Calculate approximate line numbers (not exact)
            start_line = content[:start_pos].count('\n') + 1
            end_line = start_line + section_content.count('\n')

            # Create chunk for this section
            chunk = ContentChunk(
                content=section_content,
                file_path=file_path,
                chunk_type=FILE_TYPE_DOCUMENTATION,
                start_line=start_line,
                end_line=end_line,
                language='md',
                name=heading_text,
                metadata={
                    "type": "markdown_section",
                    "heading_level": heading_level
                }
            )
            chunks.append(chunk)

        # If the first heading doesn't start at the beginning of the file,
        # add a chunk for the content before the first heading
        if headings and headings[0].start() > 0:
            prefix_content = content[:headings[0].start()]
            if prefix_content.strip():  # Only if there's non-whitespace content
                chunk = ContentChunk(
                    content=prefix_content,
                    file_path=file_path,
                    chunk_type=FILE_TYPE_DOCUMENTATION,
                    start_line=1,
                    end_line=prefix_content.count('\n') + 1,
                    language='md',
                    name="Introduction",
                    metadata={"type": "markdown_intro"}
                )
                chunks.insert(0, chunk)  # Insert at the beginning

        return chunks

    def _process_configuration_file(self, file_path: str, content: str, language: str) -> List[ContentChunk]:
        """
        Process a configuration file.

        Args:
            file_path: Path to the file
            content: File content
            language: Configuration format (e.g., json, yaml)

        Returns:
            List of ContentChunk objects
        """
        # For most configuration files, keep as a single chunk
        chunk = ContentChunk(
            content=content,
            file_path=file_path,
            chunk_type=FILE_TYPE_CONFIGURATION,
            language=language,
            metadata={"type": language}
        )

        return [chunk]

    def _chunk_by_size(
        self, content: str, file_path: str, chunk_type: str, language: str,
        chunk_size: int = 1500, overlap: int = 200
    ) -> List[ContentChunk]:
        """
        Chunk content by size with overlapping windows.

        Args:
            content: Text content to chunk
            file_path: Path to the source file
            chunk_type: Type of content
            language: Language or format
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of ContentChunk objects
        """
        chunks = []
        lines = content.splitlines()

        if not lines:
            return chunks

        chunk_text = []
        chunk_length = 0
        chunk_start_line = 1

        for i, line in enumerate(lines, 1):
            chunk_text.append(line)
            chunk_length += len(line) + 1  # +1 for the newline

            # If we've reached the target chunk size, create a chunk
            if chunk_length >= chunk_size:
                chunk_content = '\n'.join(chunk_text)
                chunk = ContentChunk(
                    content=chunk_content,
                    file_path=file_path,
                    chunk_type=chunk_type,
                    start_line=chunk_start_line,
                    end_line=i,
                    language=language,
                    metadata={"type": "size_based_chunk"}
                )
                chunks.append(chunk)

                # Calculate overlap for next chunk
                overlap_lines = []
                overlap_length = 0

                # Add lines from the end of the current chunk until we reach the overlap size
                for overlap_line in reversed(chunk_text):
                    if overlap_length + len(overlap_line) + 1 > overlap:
                        break
                    overlap_lines.insert(0, overlap_line)
                    overlap_length += len(overlap_line) + 1

                # Start the next chunk with the overlap
                chunk_text = overlap_lines
                chunk_length = overlap_length
                chunk_start_line = i - len(overlap_lines) + 1

        # Add any remaining content as a final chunk
        if chunk_text:
            final_chunk = ContentChunk(
                content='\n'.join(chunk_text),
                file_path=file_path,
                chunk_type=chunk_type,
                start_line=chunk_start_line,
                end_line=len(lines),
                language=language,
                metadata={"type": "size_based_chunk"}
            )
            chunks.append(final_chunk)

        return chunks

    def process_repository(self, exclude_dirs: Optional[List[str]] = None) -> List[ContentChunk]:
        """
        Process the entire repository.

        Args:
            exclude_dirs: List of directories to exclude (relative to repo root)

        Returns:
            List of all ContentChunk objects
        """
        self.logger.info(f"Starting repository processing: {self.repo_path}")
        start_time = time.time()

        all_chunks = []
        exclude_dirs = exclude_dirs or ['.git', 'node_modules', 'venv', 'env', '.env', 'build', 'dist']
        exclude_dirs = [os.path.normpath(d) for d in exclude_dirs]

        # Convert exclude_dirs to absolute paths for easier comparison
        exclude_abs = [os.path.join(self.repo_path, d) for d in exclude_dirs]

        # Walk through all repository files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_abs]

            # Process each file
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repo_path)

                # Process the file
                chunks = self.process_file(rel_path)
                all_chunks.extend(chunks)

        # Update final statistics
        elapsed = time.time() - start_time
        self.stats["processing_time"] = elapsed

        self.logger.info(
            f"Repository processing complete: {len(all_chunks)} chunks created from "
            f"{self.stats['files_processed']} files in {elapsed:.2f}s"
        )

        # Log detailed statistics
        self.logger.info(f"Files by type: {dict(self.stats['files_by_type'])}")
        self.logger.info(f"Chunks by type: {dict(self.stats['chunks_by_type'])}")
        self.logger.info(f"Processing errors: {self.stats['errors']}")

        return all_chunks

    def save_chunks(self, output_dir: str) -> str:
        """
        Save all chunks to a JSON file.

        Args:
            output_dir: Directory to save chunks in

        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create filename based on repository name
        repo_name = os.path.basename(self.repo_path)
        output_file = os.path.join(output_dir, f"{repo_name}_chunks.json")

        # Convert chunks to dictionaries
        chunks_data = [chunk.to_dict() for chunk in self.chunks]

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "repository": repo_name,
                "stats": self.stats,
                "chunks": chunks_data
            }, f, indent=2)

        self.logger.info(f"Saved {len(self.chunks)} chunks to {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process repository content for embedding")
    parser.add_argument("--repo-path", required=True, help="Path to the repository")
    parser.add_argument("--output-dir", default="./data", help="Directory to save chunks")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Process the repository
    processor = ContentProcessor(args.repo_path, log_level=log_level)
    chunks = processor.process_repository()
    processor.chunks = chunks

    # Save chunks to file
    output_file = processor.save_chunks(args.output_dir)
    print(f"Chunks saved to {output_file}")