#!/usr/bin/env python3
"""
RAG Engine Module

This module implements the Retrieval-Augmented Generation (RAG) engine
for answering questions about repositories. It provides functionality to:
1. Parse role-specific questions
2. Retrieve relevant content using the embeddings
3. Generate detailed answers using an LLM
4. Format answers for presentation

It implements comprehensive logging for tracking the question-answering process.
"""

import os
import json
import time
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
import importlib.util
from pathlib import Path

from embeddings_manager import EmbeddingsManager

logger = logging.getLogger("rag_engine")
logger.propagate = False
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for answering repository questions.
    """

    # Define question templates for different roles
    QUESTION_TEMPLATES = {
        "programmer": [
            "What programming languages are used in this project?",
            "What is the project's architecture/structure?",
            "What are the main components/modules of the project?",
            "What testing framework(s) are used?",
            "What dependencies does this project have?",
            "What is the code quality like (comments, documentation, etc.)?",
            "Are there any known bugs or issues?",
            "What is the build/deployment process?",
            "How is version control used in the project?",
            "What coding standards or conventions are followed?"
        ],
        "ceo": [
            "What is the overall purpose of this project?",
            "What business problem does this solve?",
            "What is the target market or user base?",
            "How mature is the project (stable, beta, etc.)?",
            "What is the competitive landscape for this project?",
            "What resources are required to maintain/develop this project?",
            "What are the potential revenue streams for this project?",
            "What are the biggest risks associated with this project?",
            "What metrics should be tracked to measure success?",
            "What is the roadmap for future development?"
        ],
        "sales_manager": [
            "What problem does this product solve for customers?",
            "What are the key features and benefits?",
            "Who is the target customer for this product?",
            "How does this product compare to competitors?",
            "What is the current state/version of the product?",
            "What are the technical requirements for using this product?",
            "Are there any case studies or success stories?",
            "What is the pricing model or strategy?",
            "What are common objections customers might have?",
            "What support options are available for customers?"
        ]
    }

    def __init__(
        self,
        embeddings_dir: str,
        repo_info: Dict[str, Any],
        use_openai: bool = False,
        use_local_llm: bool = False,
        local_llm_path: Optional[str] = None,
        local_llm_type: str = "llama2",
        log_level: int = logging.INFO
    ):
        """
        Initialize the RAG engine.

        Args:
            embeddings_dir: Directory containing embeddings and vector index
            repo_info: Dictionary with repository metadata
            use_openai: Whether to use OpenAI API for generation
            use_local_llm: Whether to use local LLM for generation
            local_llm_path: Path to local LLM model file
            local_llm_type: Type of local LLM model ('llama2' or 'codellama')
            log_level: Logging level for this engine instance
        """
        self.embeddings_dir = embeddings_dir
        self.repo_info = repo_info
        self.use_openai = use_openai
        self.use_local_llm = use_local_llm
        self.local_llm_path = local_llm_path
        self.local_llm_type = local_llm_type

        # Load embeddings manager
        self.embedding_manager = EmbeddingsManager(
            output_dir=embeddings_dir,
            log_level=log_level
        )

        # Setup logger
        self.logger = logging.getLogger(f"rag_engine.{os.path.basename(embeddings_dir)}")
        self.logger.setLevel(log_level)

        # Remove all existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler for this instance
        log_dir = os.path.join(embeddings_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rag_{int(time.time())}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Initialized RAG engine for {repo_info.get('name', 'unknown repository')}")

        # Check if OpenAI is installed
        self._has_openai = importlib.util.find_spec("openai") is not None
        if use_openai and not self._has_openai:
            self.logger.warning(
                "openai package not found but use_openai=True. "
                "Install with: pip install openai"
            )

        # Check if LocalLLM support is available
        self._has_local_llm = importlib.util.find_spec("local_llm") is not None
        if use_local_llm and not self._has_local_llm:
            self.logger.warning(
                "local_llm module not found but use_local_llm=True. "
                "Make sure local_llm.py is in your Python path."
            )

        # Load LLM
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM for generation."""
        self.openai_client = None
        self.local_llm = None

        self.logger.info(f"Initializing LLM: use_openai={self.use_openai}, use_local_llm={self.use_local_llm}")

        if self.use_openai and self._has_openai:
            try:
                import openai

                # Check for API key in environment
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    self.logger.warning(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it with: export OPENAI_API_KEY=your_key"
                    )

                self.logger.info("OpenAI client initialized")
                self.openai_client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
                self.openai_client = None

        # Add debug logs for local LLM import
        if self.use_local_llm:
            try:
                self.logger.info("Attempting to import local_llm module")
                from local_llm import LocalLLM
                self.logger.info("Local LLM module imported successfully")

                # Add checks for local LLM path
                if self.local_llm_path:
                    self.logger.info(f"Checking local LLM path: {self.local_llm_path}")
                    if os.path.exists(self.local_llm_path):
                        self.logger.info(f"Local LLM file exists: {self.local_llm_path}")

                        # Initialize the local LLM
                        self.logger.info(f"Initializing local LLM: {self.local_llm_path} (type: {self.local_llm_type})")
                        self.local_llm = LocalLLM(
                            model_path=self.local_llm_path,
                            model_type=self.local_llm_type
                        )
                    else:
                        self.logger.error(f"Local LLM file not found: {self.local_llm_path}")
                else:
                    self.logger.warning("No local LLM path provided")

                    # Try to find a default model
                    possible_paths = [
                        "../embeded_qa/llama_model/codellama-7b.Q4_K_M.gguf",
                        "../embeded_qa/models/llama-2-7b-chat.gguf"
                    ]

                    for path in possible_paths:
                        if os.path.exists(path):
                            self.logger.info(f"Found default LLM path: {path}")
                            self.local_llm_path = path
                            if "codellama" in path:
                                self.local_llm_type = "codellama"
                            else:
                                self.local_llm_type = "llama2"

                            # Initialize with the default model
                            self.logger.info(f"Initializing local LLM with default path: {self.local_llm_path}")
                            self.local_llm = LocalLLM(
                                model_path=self.local_llm_path,
                                model_type=self.local_llm_type
                            )
                            break

                    if not self.local_llm_path:
                        self.logger.error("Could not find a default LLM path")
            except ImportError as e:
                self.logger.error(f"Failed to import local_llm module: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error initializing local LLM: {e}", exc_info=True)
                self.local_llm = None

        if not (self.openai_client or self.local_llm):
            self.logger.info("Using fallback generation (no LLM)")

    def load_data(self) -> bool:
        """Load embeddings and chunks data."""
        # Load embeddings and vector DB
        embeddings_loaded = self.embedding_manager.load_embeddings()
        vector_db_loaded = self.embedding_manager.load_vector_db()

        # Get repository name and owner
        repo_name = self.repo_info.get('name', '')
        repo_owner = self.repo_info.get('owner', '')

        # Possible filename patterns
        chunk_filenames = [
            "chunks.json",                      # Generic name
            f"{repo_name}_chunks.json",         # Just repo name
            f"{repo_owner}_{repo_name}_chunks.json"  # Owner and repo name
        ]

        # Possible directory locations
        possible_dirs = [
            self.embeddings_dir,
            os.path.dirname(self.embeddings_dir),
            os.path.join(os.path.dirname(self.embeddings_dir), "data")
        ]

        # Generate all possible locations
        possible_locations = []
        for directory in possible_dirs:
            for filename in chunk_filenames:
                possible_locations.append(os.path.join(directory, filename))

        # Add any additional paths you've seen in the wild
        possible_locations.extend([
            os.path.join(os.path.dirname(self.embeddings_dir), "data", "Textualize_rich_chunks.json"),
            os.path.join(os.path.dirname(self.embeddings_dir), "data", "rich_chunks.json"),
            os.path.join(os.path.dirname(self.embeddings_dir), "data", "_chunks.json")
        ])

        # For debugging
        self.logger.info(f"Looking for chunks files with patterns: {chunk_filenames}")

        chunks_loaded = False
        for location in possible_locations:
            if os.path.exists(location):
                self.logger.info(f"Found chunks file at: {location}")
                chunks_loaded = self.embedding_manager.load_chunks(location)
                break

        if not chunks_loaded:
            locations_str = "\n - ".join(possible_locations)
            self.logger.error(f"Chunks file not found in any of these locations:\n - {locations_str}")
            return False

        if not (embeddings_loaded and vector_db_loaded):
            self.logger.error("Failed to load embeddings or vector DB")
            return False

        self.logger.info("RAG engine data loaded successfully")
        return True

    def get_questions_for_role(self, role: str) -> List[str]:
        """
        Get predefined questions for a specific role.

        Args:
            role: Role (programmer, ceo, sales_manager)

        Returns:
            List of questions for the role
        """
        if role.lower() not in self.QUESTION_TEMPLATES:
            self.logger.warning(f"Unknown role: {role}, using programmer as default")
            return self.QUESTION_TEMPLATES["programmer"]

        return self.QUESTION_TEMPLATES[role.lower()]

    def retrieve_relevant_chunks(
        self, question: str, top_k: int = 5, threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to the question.

        Args:
            question: Question text
            top_k: Number of chunks to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of relevant chunks with metadata
        """
        # Search for similar chunks
        results = self.embedding_manager.search_similar_chunks(question, top_k=top_k)

        # Filter by similarity threshold
        filtered_results = [r for r in results if r["similarity"] >= threshold]

        self.logger.info(
            f"Retrieved {len(filtered_results)} chunks for question: "
            f"\"{question[:50]}{'...' if len(question) > 50 else ''}\""
        )

        return filtered_results

    def _generate_answer_with_llm(
        self, question: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> str:
        """
        Generate answer using an LLM.

        Args:
            question: Question text
            chunks: Relevant chunks
            max_tokens: Maximum tokens for generation

        Returns:
            Generated answer
        """
        self.logger.info(f"Generating answer with LLM. Available: OpenAI={self.openai_client is not None}, Local LLM={self.local_llm is not None}")

        # Try OpenAI first if configured
        if self.openai_client:
            self.logger.info("Using OpenAI for generation")
            return self._generate_with_openai(question, chunks, max_tokens)

        # Try local LLM if available
        if self.local_llm:
            self.logger.info("Using local LLM for generation")
            return self._generate_with_local_llm(question, chunks, max_tokens)

        # Fall back to basic generation
        self.logger.warning("No LLM available, using fallback generation")
        return self._generate_answer_fallback(question, chunks)

    def _generate_with_openai(
        self, question: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> str:
        """Generate answer using OpenAI API."""
        try:
            # Import tiktoken for token counting
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

            # Prepare repository information
            repo_name = self.repo_info.get('name', 'Unknown')
            repo_owner = self.repo_info.get('owner', 'Unknown')
            repo_type = self.repo_info.get('type', 'Unknown')
            repo_languages = ", ".join(self.repo_info.get('languages', {}).keys())

            # Create base prompt
            system_prompt = f"""
            You are a technical analyst providing information about the repository {repo_name} by {repo_owner}.
            Answer the question thoroughly based ONLY on the provided context.
            If the context doesn't contain enough information, say so clearly.
            """

            # Count tokens in the system prompt
            system_tokens = len(encoding.encode(system_prompt))
            question_tokens = len(encoding.encode(question))

            # Set maximum context tokens (leaving room for response)
            max_context_tokens = 8000 - system_tokens - question_tokens - max_tokens

            # Prepare context from chunks, limiting to max_context_tokens
            context = []
            current_tokens = 0

            # Sort chunks by similarity (assuming they're already sorted)
            for i, chunk_data in enumerate(chunks[:10], 1):  # Limit to top 10 chunks max
                chunk = chunk_data["chunk"]
                chunk_text = f"CONTEXT {i}:\nFile: {chunk['file_path']}"
                if chunk.get('name'):
                    chunk_text += f"\nName: {chunk['name']}"

                # Truncate content if needed
                content = chunk['content']
                if len(content) > 2000:  # Arbitrary limit to avoid extremely long chunks
                    content = content[:2000] + "... [truncated]"

                chunk_text += f"\nContent:\n{content}\n"

                # Count tokens for this chunk
                chunk_tokens = len(encoding.encode(chunk_text))

                # Add chunk if it fits within the token limit
                if current_tokens + chunk_tokens <= max_context_tokens:
                    context.append(chunk_text)
                    current_tokens += chunk_tokens
                else:
                    self.logger.info(f"Skipping chunk {i} due to token limit")
                    break

            context_text = "\n".join(context)

            # Final prompt
            user_prompt = f"""
            Question: {question}
            
            Here is the relevant information from the repository:
            
            {context_text}
            
            Answer the question based on this information. Be specific and reference relevant files or code.
            """

            # Calculate final token count
            total_input_tokens = system_tokens + len(encoding.encode(user_prompt))
            self.logger.info(f"Total input tokens: {total_input_tokens}")

            # If still too large, further reduce
            if total_input_tokens > 15000:  # OpenAI's limit is higher, but be safe
                self.logger.warning(f"Input still too large ({total_input_tokens} tokens), reducing further")
                return self._generate_answer_fallback(question, chunks)

            # Call OpenAI API
            start_time = time.time()

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller model to reduce token costs
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2
            )

            elapsed = time.time() - start_time

            # Extract answer from response
            answer = response.choices[0].message.content

            # Log token usage
            usage = response.usage
            self.logger.info(
                f"Token usage - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}"
            )

            self.logger.info(
                f"Generated answer with OpenAI in {elapsed:.2f}s, "
                f"{len(answer)} chars"
            )

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer with OpenAI: {e}", exc_info=True)
            return self._generate_answer_fallback(question, chunks)
    def _generate_with_local_llm(
        self, question: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> str:
        """
        Generate answer using local LLM.

        Args:
            question: Question text
            chunks: Relevant chunks
            max_tokens: Maximum tokens for generation

        Returns:
            Generated answer
        """
        try:
            # Prepare context from chunks
            context_parts = []
            for i, chunk_data in enumerate(chunks, 1):
                chunk = chunk_data["chunk"]
                context_parts.append(f"[Document {i}]")
                context_parts.append(f"File: {chunk['file_path']}")
                if chunk.get('name'):
                    context_parts.append(f"Name: {chunk['name']}")
                context_parts.append(f"Content:\n{chunk['content']}\n")

            context_text = "\n".join(context_parts)

            # Generate answer with local LLM
            start_time = time.time()

            self.logger.info(f"Generating answer with local LLM for question: {question[:50]}...")

            answer = self.local_llm.generate_answer(
                question=question,
                context=context_text,
                max_tokens=max_tokens,
                temperature=0.2
            )

            elapsed = time.time() - start_time

            self.logger.info(
                f"Generated answer with local LLM in {elapsed:.2f}s, "
                f"{len(answer)} chars"
            )

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer with local LLM: {e}", exc_info=True)
            return self._generate_answer_fallback(question, chunks)

    def _generate_answer_fallback(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a simple answer without LLM when OpenAI is not available.

        Args:
            question: Question text
            chunks: Relevant chunks

        Returns:
            Generated answer
        """
        # Create a basic response from the chunks
        lines = [f"Answer to: {question}\n"]
        lines.append("Based on the repository information:\n")

        for i, chunk_data in enumerate(chunks[:3], 1):  # Limit to top 3 chunks
            chunk = chunk_data["chunk"]
            lines.append(f"Source {i}: {chunk['file_path']}")

            if chunk.get('name'):
                lines.append(f"Name: {chunk['name']}")

            # Truncate content
            content = chunk['content']
            if len(content) > 300:
                content = content[:300] + "..."

            lines.append(f"Content excerpt:\n{content}\n")

        lines.append("\nNote: This is a simplified answer without using an LLM. For more comprehensive answers, please set up OpenAI API access.")

        return "\n".join(lines)

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question about the repository.

        Args:
            question: Question text
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with question, answer, and supporting chunks
        """
        start_time = time.time()
        self.logger.info(f"Processing question: {question}")

        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(question, top_k=top_k)

        if not chunks:
            answer = "I couldn't find relevant information in the repository to answer this question."
            self.logger.warning(f"No relevant chunks found for question: {question}")
        else:
            # Generate answer
            answer = self._generate_answer_with_llm(question, chunks)

        elapsed = time.time() - start_time

        # Format result
        result = {
            "question": question,
            "answer": answer,
            "supporting_chunks": [c["chunk"] for c in chunks],
            "processing_time": elapsed
        }

        self.logger.info(f"Answered question in {elapsed:.2f}s")
        return result

    def process_role_questions(self, role: str) -> List[Dict[str, Any]]:
        """
        Process all questions for a specific role.

        Args:
            role: Role (programmer, ceo, sales_manager)

        Returns:
            List of question-answer dictionaries
        """
        questions = self.get_questions_for_role(role)

        self.logger.info(f"Processing {len(questions)} questions for role: {role}")
        start_time = time.time()

        results = []
        for i, question in enumerate(questions, 1):
            self.logger.info(f"Question {i}/{len(questions)}: {question}")

            result = self.answer_question(question)
            results.append(result)

        elapsed = time.time() - start_time
        self.logger.info(f"Processed all {len(questions)} questions in {elapsed:.2f}s")

        return results

    def generate_report_data(self, role: str) -> Dict[str, Any]:
        """
        Generate data for a report.

        Args:
            role: Role (programmer, ceo, sales_manager)

        Returns:
            Dictionary with report data
        """
        # Process role questions
        qa_results = self.process_role_questions(role)

        # Format report data
        report_data = {
            "repository": self.repo_info,
            "role": role,
            "qa_pairs": qa_results,
            "generation_time": sum(r["processing_time"] for r in qa_results),
            "timestamp": time.time()
        }

        # Save report data to file
        output_dir = os.path.join(self.embeddings_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)

        repo_name = self.repo_info.get('name', 'unknown')
        output_file = os.path.join(
            output_dir,
            f"{repo_name}_{role}_report_{int(time.time())}.json"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Report data saved to {output_file}")

        return report_data


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer questions about repositories using RAG")
    parser.add_argument("--embeddings-dir", required=True, help="Directory containing embeddings")
    parser.add_argument("--repo-info", required=True, help="JSON file with repository info")
    parser.add_argument("--role", default="programmer", choices=["programmer", "ceo", "sales_manager"],
                      help="Role perspective for analysis")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for generation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--question", help="Single question to answer (optional)")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Load repo info
    with open(args.repo_info, 'r', encoding='utf-8') as f:
        repo_info = json.load(f)

    # Initialize RAG engine
    rag_engine = RAGEngine(
        embeddings_dir=dirs["embeddings"],
        repo_info=repo_info,
        use_openai=args.use_openai,
        use_local_llm=args.use_local_llm,
        local_llm_path=args.local_llm_path,
        local_llm_type=args.local_llm_type,
        log_level=log_level
    )


    # Load data
    if not engine.load_data():
        print("Failed to load data. Exiting.")
        exit(1)

    if args.question:
        # Answer single question
        result = engine.answer_question(args.question)

        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nBased on {len(result['supporting_chunks'])} chunks in {result['processing_time']:.2f}s")
    else:
        # Process all questions for the role
        report_data = engine.generate_report_data(args.role)

        print(f"\nGenerated report data for {args.role} role")
        print(f"Repository: {report_data['repository']['name']}")
        print(f"Processed {len(report_data['qa_pairs'])} questions")
        print(f"Total processing time: {report_data['generation_time']:.2f}s")