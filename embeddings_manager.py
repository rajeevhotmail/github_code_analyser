#!/usr/bin/env python3
"""
Embeddings Manager Module

This module handles the creation and management of embeddings for repository content.
It provides functionality to:
1. Generate embeddings for content chunks
2. Store embeddings in vector database
3. Retrieve relevant chunks based on queries
4. Manage different embedding models and strategies

It implements comprehensive logging for tracking embedding generation and retrieval.
"""

import os
import json
import time
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import importlib.util

# Setup module logger
logger = logging.getLogger("embeddings_manager")
logger.propagate = False
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class EmbeddingsManager:
    """
    Manages embeddings for repository content chunks.
    Handles embedding generation, storage, and retrieval.
    """

    def __init__(
        self,
        output_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
        log_level: int = logging.INFO
    ):
        """
        Initialize the embeddings manager.

        Args:
            output_dir: Directory to store embeddings and index
            model_name: Name of the sentence-transformer model to use
            use_gpu: Whether to use GPU for embedding generation
            log_level: Logging level for this manager instance
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.embeddings = {}
        self.chunks = []
        self.vector_db = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Setup processor-specific logger
        self.logger = logging.getLogger(f"embeddings_manager.{os.path.basename(output_dir)}")
        self.logger.setLevel(log_level)

        # Create file handler for this instance
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"embeddings_{int(time.time())}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Initialized embeddings manager with model {model_name}")

        # Check if sentence-transformers is installed
        self._check_dependencies()

        # Initialize embeddings model
        if self._has_sentence_transformers:
            self._init_model()

    def _check_dependencies(self):
        """Check if necessary dependencies are installed."""
        self._has_sentence_transformers = importlib.util.find_spec("sentence_transformers") is not None
        self._has_faiss = importlib.util.find_spec("faiss") is not None

        if not self._has_sentence_transformers:
            self.logger.warning(
                "sentence-transformers package not found. Please install with: "
                "pip install sentence-transformers"
            )

        if not self._has_faiss:
            self.logger.warning(
                "faiss-cpu package not found. Vector search will not be available. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )

    def _init_model(self):
        """Initialize the embedding model."""
        if not self._has_sentence_transformers:
            self.logger.error("Cannot initialize model: sentence-transformers not installed")
            return

        try:
            start_time = time.time()
            from sentence_transformers import SentenceTransformer

            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

            elapsed = time.time() - start_time
            self.logger.info(f"Loaded model {self.model_name} in {elapsed:.2f}s on {device}")
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}", exc_info=True)
            self.model = None

    def load_chunks(self, chunks_file: str) -> int:
        """
        Load content chunks from a JSON file.

        Args:
            chunks_file: Path to the JSON chunks file

        Returns:
            Number of chunks loaded
        """
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.chunks = data["chunks"]

            self.logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
            return len(self.chunks)
        except Exception as e:
            self.logger.error(f"Error loading chunks from {chunks_file}: {e}", exc_info=True)
            return 0

    def _preprocess_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Preprocess a chunk for embedding.

        Args:
            chunk: Content chunk dictionary

        Returns:
            Preprocessed text suitable for embedding
        """
        content = chunk["content"]

        # Add metadata as prefix for better context
        prefix = []

        if chunk["file_path"]:
            prefix.append(f"File: {chunk['file_path']}")

        if chunk["chunk_type"]:
            prefix.append(f"Type: {chunk['chunk_type']}")

        if chunk["language"]:
            prefix.append(f"Language: {chunk['language']}")

        if chunk["name"]:
            prefix.append(f"Name: {chunk['name']}")

        if chunk["parent"]:
            prefix.append(f"Parent: {chunk['parent']}")

        # Join prefix with content
        if prefix:
            prefix_text = " | ".join(prefix)
            return f"{prefix_text}\n\n{content}"
        else:
            return content

    def generate_embeddings(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Generate embeddings for all loaded chunks.

        Args:
            batch_size: Number of chunks to process in each batch

        Returns:
            Dictionary with embedding statistics
        """
        if not self.model:
            self.logger.error("Cannot generate embeddings: model not initialized")
            return {"error": "Model not initialized", "chunks_processed": 0}

        if not self.chunks:
            self.logger.warning("No chunks loaded, nothing to embed")
            return {"error": "No chunks loaded", "chunks_processed": 0}

        self.logger.info(f"Generating embeddings for {len(self.chunks)} chunks")
        start_time = time.time()

        # Prepare chunks for embedding
        chunk_texts = []
        for chunk in self.chunks:
            processed_text = self._preprocess_chunk(chunk)
            chunk_texts.append(processed_text)

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]

            batch_start = time.time()
            self.logger.debug(f"Processing batch {i//batch_size + 1}/{len(chunk_texts)//batch_size + 1}")

            try:
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)

                batch_time = time.time() - batch_start
                self.logger.debug(f"Batch processed in {batch_time:.2f}s ({len(batch)/batch_time:.1f} chunks/s)")
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                # Continue with next batch

        # Concatenate all embeddings
        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)

            # Add chunk IDs
            self.embeddings = {
                "ids": [i for i in range(len(self.chunks))],
                "vectors": embeddings_array,
                "model": self.model_name,
                "dimensions": embeddings_array.shape[1]
            }

            elapsed = time.time() - start_time
            self.logger.info(
                f"Generated {len(self.chunks)} embeddings "
                f"({embeddings_array.shape[1]} dimensions) in {elapsed:.2f}s"
            )

            # Save embeddings
            self._save_embeddings()

            # Create vector database
            self._create_vector_db()

            return {
                "chunks_processed": len(self.chunks),
                "embedding_dimensions": embeddings_array.shape[1],
                "processing_time": elapsed
            }
        else:
            self.logger.error("Failed to generate any embeddings")
            return {"error": "Failed to generate embeddings", "chunks_processed": 0}

    def _save_embeddings(self):
        """Save embeddings to disk."""
        if not self.embeddings:
            self.logger.warning("No embeddings to save")
            return

        # Save embeddings array
        embeddings_file = os.path.join(self.output_dir, "embeddings.pkl")
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)

            self.logger.info(f"Saved embeddings to {embeddings_file}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}", exc_info=True)

    def load_embeddings(self) -> bool:
        """
        Load embeddings from disk.

        Returns:
            True if embeddings were loaded successfully, False otherwise
        """
        embeddings_file = os.path.join(self.output_dir, "embeddings.pkl")

        if not os.path.exists(embeddings_file):
            self.logger.warning(f"Embeddings file not found: {embeddings_file}")
            return False

        try:
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

            self.logger.info(
                f"Loaded embeddings: {len(self.embeddings['ids'])} vectors, "
                f"{self.embeddings['dimensions']} dimensions"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}", exc_info=True)
            return False

    def _create_vector_db(self):
        """Create a vector database for similarity search."""
        if not self._has_faiss:
            self.logger.warning("Cannot create vector DB: faiss not installed")
            return

        if not self.embeddings or "vectors" not in self.embeddings:
            self.logger.warning("No embeddings available to create vector DB")
            return

        try:
            import faiss

            # Get embeddings array
            vectors = self.embeddings["vectors"].astype(np.float32)
            dimension = vectors.shape[1]

            # Create FAISS index
            start_time = time.time()

            # Use L2 distance for similarity
            index = faiss.IndexFlatL2(dimension)

            # Add vectors to index
            index.add(vectors)

            elapsed = time.time() - start_time
            self.logger.info(
                f"Created FAISS index with {vectors.shape[0]} vectors "
                f"in {elapsed:.2f}s"
            )

            # Save vector database
            index_file = os.path.join(self.output_dir, "faiss_index.bin")
            faiss.write_index(index, index_file)
            self.logger.info(f"Saved FAISS index to {index_file}")

            self.vector_db = index
        except Exception as e:
            self.logger.error(f"Error creating vector DB: {e}", exc_info=True)

    def load_vector_db(self) -> bool:
        """
        Load vector database from disk.

        Returns:
            True if vector DB was loaded successfully, False otherwise
        """
        if not self._has_faiss:
            self.logger.warning("Cannot load vector DB: faiss not installed")
            return False

        index_file = os.path.join(self.output_dir, "faiss_index.bin")

        if not os.path.exists(index_file):
            self.logger.warning(f"Vector DB file not found: {index_file}")
            return False

        try:
            import faiss

            start_time = time.time()
            self.vector_db = faiss.read_index(index_file)

            elapsed = time.time() - start_time
            self.logger.info(
                f"Loaded FAISS index with {self.vector_db.ntotal} vectors "
                f"in {elapsed:.2f}s"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading vector DB: {e}", exc_info=True)
            return False

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of dictionaries with similar chunks and similarity scores
        """
        if not self.model:
            self.logger.error("Cannot search: model not initialized")
            return []

        if not self.vector_db:
            self.logger.error("Cannot search: vector DB not initialized")
            return []

        if not self.chunks:
            self.logger.error("Cannot search: no chunks available")
            return []

        try:
            start_time = time.time()

            # Generate embedding for query
            query_embedding = self.model.encode([query])[0].astype(np.float32)
            query_embedding = query_embedding.reshape(1, -1)

            # Search in vector DB
            distances, indices = self.vector_db.search(query_embedding, top_k)

            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.chunks):
                    continue  # Skip invalid indices

                chunk = self.chunks[idx]
                results.append({
                    "chunk": chunk,
                    "similarity": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "distance": float(distance),
                    "rank": i + 1
                })

            elapsed = time.time() - start_time
            self.logger.info(f"Search completed in {elapsed:.4f}s, found {len(results)} results")

            return results
        except Exception as e:
            self.logger.error(f"Error searching vector DB: {e}", exc_info=True)
            return []


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and manage embeddings for repository content")
    parser.add_argument("--chunks-file", required=True, help="Path to the chunks JSON file")
    parser.add_argument("--output-dir", default="./embeddings", help="Directory to save embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for embedding generation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--query", help="Test query to search for similar chunks")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Initialize embeddings manager
    manager = EmbeddingsManager(
        output_dir=args.output_dir,
        model_name=args.model,
        use_gpu=args.use_gpu,
        log_level=log_level
    )

    # Load chunks
    if not os.path.exists(args.chunks_file):
        print(f"Chunks file not found: {args.chunks_file}")
        exit(1)

    manager.load_chunks(args.chunks_file)

    # Generate embeddings
    stats = manager.generate_embeddings()
    print(f"Embedding stats: {stats}")

    # Test search if query provided
    if args.query:
        print(f"\nSearching for: {args.query}")
        results = manager.search_similar_chunks(args.query, top_k=5)

        print(f"Found {len(results)} results:")
        for result in results:
            chunk = result["chunk"]
            print(f"  Rank {result['rank']} - Similarity: {result['similarity']:.4f}")
            print(f"  File: {chunk['file_path']}")
            if chunk['name']:
                print(f"  Name: {chunk['name']}")
            print(f"  Type: {chunk['chunk_type']}/{chunk['language']}")
            print(f"  Content preview: {chunk['content'][:100]}...")
            print()