#!/usr/bin/env python3
"""
Repository Analysis Tool

This script analyzes GitHub or GitLab repositories based on a specified role
and provides answers to predefined questions in a PDF document.

Usage:
    python main_repo_analyst.py --url <repository_url> --role <role> [--persistent]
    python main_repo_analyst.py --local-path <repository_path> --role <role>

Example:
    python main_repo_analyst.py --url https://github.com/fastapi-users/fastapi-users --role programmer --persistent
"""

import argparse
import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from repo_fetcher import RepoFetcher
from content_processor import ContentProcessor
from embeddings_manager import EmbeddingsManager
from rag_engine import RAGEngine
from pdf_generator import PDFGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")
logger.propagate = False
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Create file handler for main script
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"repo_analysis_{int(time.time())}.log")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GitHub/GitLab repositories")
    parser.add_argument("--url", help="Repository URL")
    parser.add_argument("--local-path", help="Path to local repository")
    parser.add_argument("--role", required=True, choices=["programmer", "ceo", "sales_manager"],
                       help="Role perspective for analysis")
    parser.add_argument("--github-token", help="GitHub personal access token")
    parser.add_argument("--gitlab-token", help="GitLab personal access token")
    parser.add_argument("--output-dir", default="./output", help="Base directory for outputs")
    parser.add_argument("--persistent", action="store_true", help="Clone to a persistent directory")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for answer generation")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip repository fetching (use existing)")
    parser.add_argument("--skip-process", action="store_true", help="Skip content processing (use existing)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation (use existing)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Ensure at least one of --url or --local-path is provided
    if not args.url and not args.local_path and not (args.skip_fetch and args.skip_process):
        parser.error("Either --url or --local-path is required")

    return args


def setup_directories(base_dir, repo_name):
    """
    Set up output directories for processing.

    Args:
        base_dir: Base output directory
        repo_name: Repository name

    Returns:
        Dictionary of path configurations
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    # Create repository-specific directories
    repo_dir = os.path.join(base_dir, repo_name)
    os.makedirs(repo_dir, exist_ok=True)

    data_dir = os.path.join(repo_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    embeddings_dir = os.path.join(repo_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    reports_dir = os.path.join(repo_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    return {
        "base": base_dir,
        "repo": repo_dir,
        "data": data_dir,
        "embeddings": embeddings_dir,
        "reports": reports_dir
    }


def main():
    """Main entry point for the script."""
    start_time = time.time()
    args = parse_arguments()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)

    try:
        logger.info("Starting repository analysis")

        # Step 1: Fetch repository
        if not args.skip_fetch:
            logger.info("Fetching repository...")

            repo_source = args.url if args.url else args.local_path
            source_type = "URL" if args.url else "local path"
            logger.info(f"Repository source: {source_type} {repo_source}")

            with RepoFetcher(github_token=args.github_token, gitlab_token=args.gitlab_token) as fetcher:
                repo_path = fetcher.fetch_repo(
                    url=args.url if args.url else "local",
                    local_path=args.local_path,
                    persistent=args.persistent
                )
                logger.info(f"Repository accessed at: {repo_path}")

                # Get basic repository information
                repo_info = fetcher.get_basic_repo_info()

                # Setup directories
                repo_name = repo_info['name']
                dirs = setup_directories(args.output_dir, repo_name)

                # Save repository info
                repo_info_file = os.path.join(dirs["repo"], "repo_info.json")
                with open(repo_info_file, "w", encoding="utf-8") as f:
                    json.dump(repo_info, f, indent=2)
                logger.info(f"Repository info saved to {repo_info_file}")
        else:
            # Find existing repository info
            if args.local_path:
                repo_name = os.path.basename(os.path.normpath(args.local_path))
            else:
                # Try to guess repo name from URL
                repo_name = args.url.rstrip("/").split("/")[-1] if args.url else "unknown"

            dirs = setup_directories(args.output_dir, repo_name)

            repo_info_file = os.path.join(dirs["repo"], "repo_info.json")
            if os.path.exists(repo_info_file):
                with open(repo_info_file, "r", encoding="utf-8") as f:
                    repo_info = json.load(f)
                logger.info(f"Loaded existing repository info from {repo_info_file}")
                repo_path = args.local_path
            else:
                logger.error("Repository info not found and --skip-fetch is set")
                sys.exit(1)

        # Step 2: Process content
        if not args.skip_process:
            logger.info("Processing repository content...")

            processor = ContentProcessor(repo_path, log_level=log_level)
            chunks = processor.process_repository()
            processor.chunks = chunks

            # Save chunks
            chunks_file = processor.save_chunks(dirs["data"])
            logger.info(f"Content chunks saved to {chunks_file}")
        else:
            # Find existing chunks file
            chunks_file = os.path.join(dirs["data"], f"{repo_info['name']}_chunks.json")
            if not os.path.exists(chunks_file):
                logger.error(f"Chunks file not found: {chunks_file} and --skip-process is set")
                sys.exit(1)
            logger.info(f"Using existing chunks file: {chunks_file}")

        # Step 3: Generate embeddings
        if not args.skip_embed:
            logger.info("Generating embeddings...")

            embeddings_manager = EmbeddingsManager(
                output_dir=dirs["embeddings"],
                log_level=log_level
            )

            # Load chunks
            embeddings_manager.load_chunks(chunks_file)

            # Generate embeddings
            embedding_stats = embeddings_manager.generate_embeddings()
            logger.info(f"Embeddings generated: {embedding_stats}")

        # Step 4: Answer questions with RAG
        logger.info(f"Answering questions for {args.role} role...")

        rag_engine = RAGEngine(
            embeddings_dir=dirs["embeddings"],
            repo_info=repo_info,
            use_openai=args.use_openai,
            log_level=log_level
        )

        # Load data
        if not rag_engine.load_data():
            logger.error("Failed to load data for RAG engine")
            sys.exit(1)

        # Generate report data
        report_data = rag_engine.generate_report_data(args.role)
        logger.info(f"Generated report data for {args.role} role")

        # Step 5: Generate PDF
        logger.info("Generating PDF report...")

        pdf_generator = PDFGenerator(
            output_dir=dirs["reports"],
            log_level=log_level
        )

        pdf_file = pdf_generator.generate_pdf(report_data)

        if pdf_file:
            logger.info(f"PDF report generated: {pdf_file}")
            print(f"\nAnalysis complete! PDF report generated: {pdf_file}")
        else:
            logger.error("Failed to generate PDF report")
            print("\nAnalysis complete, but PDF generation failed.")

        # Report timing
        elapsed = time.time() - start_time
        logger.info(f"Repository analysis completed in {elapsed:.2f}s")
        print(f"Total processing time: {elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"Error analyzing repository: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Repository Analysis Tool

This script analyzes GitHub or GitLab repositories based on a specified role
and provides answers to predefined questions.

Usage:
    python main_repo_analyst.py --url <repository_url> --role <role> [--persistent]
    python main_repo_analyst.py --local-path <repository_path> --role <role>

Example:
    python main_repo_analyst.py --url https://github.com/fastapi-users/fastapi-users --role programmer --persistent
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

from repo_fetcher import RepoFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define questions for each role
QUESTIONS = {
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GitHub/GitLab repositories")
    parser.add_argument("--url", help="Repository URL")
    parser.add_argument("--local-path", help="Path to local repository")
    parser.add_argument("--role", required=True, choices=["programmer", "ceo", "sales_manager"],
                        help="Role perspective for analysis")
    parser.add_argument("--github-token", help="GitHub personal access token")
    parser.add_argument("--gitlab-token", help="GitLab personal access token")
    parser.add_argument("--output-dir", default="./output", help="Base directory for outputs")
    parser.add_argument("--persistent", action="store_true", help="Clone to a persistent directory")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for answer generation")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip repository fetching (use existing)")
    parser.add_argument("--skip-process", action="store_true", help="Skip content processing (use existing)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation (use existing)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Ensure at least one of --url or --local-path is provided
    if not args.url and not args.local_path and not (args.skip_fetch and args.skip_process):
        parser.error("Either --url or --local-path is required")

    return args

def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Arguments have been validated in parse_arguments()

    try:
        repo_source = args.url if args.url else args.local_path
        source_type = "URL" if args.url else "local path"
        logger.info(f"Analyzing repository from {source_type}: {repo_source} with {args.role} perspective")

        # Fetch repository
        with RepoFetcher(github_token=args.github_token, gitlab_token=args.gitlab_token) as fetcher:
            logger.info("Accessing repository...")
            repo_path = fetcher.fetch_repo(
                url=args.url if args.url else "local",
                local_path=args.local_path,
                persistent=args.persistent
            )
            logger.info(f"Repository accessed at: {repo_path}")

            # Get basic repository information
            logger.info("Gathering repository metadata...")
            repo_info = fetcher.get_basic_repo_info()

            # Print basic repository information
            print("\nRepository Information:")
            print(f"  Name: {repo_info['name']}")
            print(f"  Owner: {repo_info['owner']}")
            print(f"  Type: {repo_info['type']}")
            print(f"  URL: {repo_info['url']}")
            print(f"  Commits: {repo_info['commit_count']}")
            print(f"  Contributors: {repo_info['contributor_count']}")
            print(f"  Default branch: {repo_info['default_branch']}")
            print(f"  Languages: {', '.join(repo_info['languages'].keys())}")

            # Get questions for the selected role
            questions = QUESTIONS[args.role]
            print(f"\nAnalyzing from {args.role.upper()} perspective:")
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")

            # This is where we would process the repository data to answer questions
            # For now, we're just showing the basic structure
            print("\nRepository files (up to 10):")
            files = fetcher.get_file_list(max_files=10)
            for file in files:
                print(f"  {file}")

            # This is where we would generate the PDF report
            # For now, just let the user know
            print(f"\nIn a complete implementation, a PDF report would be generated as {args.output}")

    except Exception as e:
        logger.error(f"Error analyzing repository: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()