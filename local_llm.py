#!/usr/bin/env python3
"""
Local LLM Module

This module provides interfaces to use local LLaMA models as LLMs for the RAG pipeline.
It supports:
1. Loading and running local GGUF models using llama-cpp-python
2. Formatting context and questions appropriately for different models
3. Managing generation parameters for optimal results
"""

import os
import time
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Union

# Setup module logger
logger = logging.getLogger("local_llm")
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class LocalLLM:
    """
    Interface for local LLaMA models.
    Supports different model formats and configurations.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "llama2",
        context_window: int = 4096,
        log_level: int = logging.INFO
    ):
        """
        Initialize the local LLM.

        Args:
            model_path: Path to the GGUF model file
            model_type: Type of model ('llama2' or 'codellama')
            context_window: Context window size for the model
            log_level: Logging level for this LLM instance
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.context_window = context_window

        # Setup logger
        self.logger = logging.getLogger(f"local_llm.{os.path.basename(model_path)}")
        self.logger.setLevel(log_level)

        # Check if llama-cpp-python is installed
        self._has_llama_cpp = importlib.util.find_spec("llama_cpp") is not None
        if not self._has_llama_cpp:
            self.logger.warning(
                "llama_cpp package not found. Install with: pip install llama-cpp-python"
            )

        # Load the model
        self.llm = None
        self._load_model()

    def _load_model(self):
        """Load the LLaMA model."""
        if not self._has_llama_cpp:
            self.logger.error("Cannot load model: llama_cpp not installed")
            return

        try:
            import llama_cpp

            start_time = time.time()
            self.logger.info(f"Loading model from {self.model_path}")

            # Load the model with appropriate parameters
            self.llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.context_window,
                n_threads=os.cpu_count() or 4,  # Use all available threads
            )

            elapsed = time.time() - start_time
            self.logger.info(f"Model loaded in {elapsed:.2f}s")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            self.llm = None

    def _format_prompt(self, question: str, context: str) -> str:
        """
        Format prompt based on model type.

        Args:
            question: The question to answer
            context: Context information

        Returns:
            Formatted prompt
        """
        if self.model_type == "llama2":
            # Use Llama 2 chat format
            system_prompt = (
                "You are a technical analyst providing information about a code repository. "
                "Answer the question based ONLY on the provided context. "
                "If the context doesn't contain enough information, say so clearly. "
                "Be specific and reference relevant files or code from the context."
            )

            # Llama 2 chat uses a specific format with <s>, [INST], etc.
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"

        elif self.model_type == "codellama":
            # Use CodeLlama format (similar to Llama 2 but with programming focus)
            system_prompt = (
                "You are a code analysis assistant. "
                "Answer the question about the repository based ONLY on the provided context. "
                "If the context doesn't contain enough information, say so clearly. "
                "Be specific and reference relevant files or code from the context."
            )

            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"

        else:
            # Generic format for other models
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> str:
        """
        Generate an answer using the local LLM.

        Args:
            question: Question to answer
            context: Context information from repository
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (higher = more creative)

        Returns:
            Generated answer text
        """
        if not self.llm:
            self.logger.error("Cannot generate answer: model not loaded")
            return "Error: Model not loaded. Please check logs for details."

        try:
            # Format the prompt
            prompt = self._format_prompt(question, context)

            start_time = time.time()
            self.logger.info(f"Generating answer for question: {question[:50]}{'...' if len(question) > 50 else ''}")

            # Generate answer
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["</s>", "[INST]", "[/INST]"],  # Stop tokens depend on the model format
            )

            # Extract the generated text
            answer = output['choices'][0]['text']

            elapsed = time.time() - start_time
            tokens_generated = len(answer.split())
            self.logger.info(f"Answer generated in {elapsed:.2f}s ({tokens_generated} tokens)")

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer: {e}", exc_info=True)
            return f"Error generating answer. Please check logs for details."


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test local LLM generation")
    parser.add_argument("--model-path", required=True, help="Path to the GGUF model file")
    parser.add_argument("--model-type", default="llama2", choices=["llama2", "codellama"],
                      help="Type of model")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--context-file", help="File containing context (optional)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Load context
    if args.context_file and os.path.exists(args.context_file):
        with open(args.context_file, 'r', encoding='utf-8') as f:
            context = f.read()
    else:
        context = "No context provided."

    # Initialize LLM
    llm = LocalLLM(
        model_path=args.model_path,
        model_type=args.model_type,
        log_level=log_level
    )

    # Generate answer
    answer = llm.generate_answer(args.question, context)

    print("\nQuestion:", args.question)
    print("\nAnswer:", answer)