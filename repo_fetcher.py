#!/usr/bin/env python3
"""
Repository Fetcher Module

This module handles cloning and basic analysis of GitHub and GitLab repositories.
It provides functionality to:
1. Clone repositories
2. Extract basic metadata
3. Handle authentication for private repositories
"""

import os
import re
import tempfile
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import requests
from git import Repo, GitCommandError, InvalidGitRepositoryError
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("repo_fetcher")
logger.propagate = False

if not logger.handlers:  # <-- This check is important
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class RepoFetcher:
    """Class to handle fetching and basic analysis of Git repositories."""

    def __init__(self, github_token: Optional[str] = None, gitlab_token: Optional[str] = None):
        """
        Initialize the RepoFetcher with optional authentication tokens.

        Args:
            github_token: Personal access token for GitHub
            gitlab_token: Personal access token for GitLab
        """
        self.github_token = github_token
        self.gitlab_token = gitlab_token
        self.temp_dir = None
        self.repo_path = None
        self.repo_url = None
        self.repo_type = None  # 'github' or 'gitlab'
        self.repo_name = None
        self.repo_owner = None

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory on exit."""
        self.cleanup()

    def cleanup(self):
        """Remove temporary directory and all its contents."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            self.repo_path = None

    def _parse_repo_url(self, url: str) -> Tuple[str, str, str]:
        """
        Parse repository URL to extract repository type, owner, and name.

        Args:
            url: Repository URL (GitHub or GitLab)

        Returns:
            Tuple containing (repo_type, owner, name)

        Raises:
            ValueError: If URL format is not recognized
        """
        # GitHub URL patterns
        github_patterns = [
            r'https://github\.com/([^/]+)/([^/]+)(?:\.git)?/?$',
            r'git@github\.com:([^/]+)/([^/]+)(?:\.git)?/?$'
        ]

        # GitLab URL patterns
        gitlab_patterns = [
            r'https://gitlab\.com/([^/]+)/([^/]+)(?:\.git)?/?$',
            r'git@gitlab\.com:([^/]+)/([^/]+)(?:\.git)?/?$'
        ]

        # Try GitHub patterns
        for pattern in github_patterns:
            match = re.match(pattern, url)
            if match:
                owner, name = match.groups()
                return 'github', owner, name

        # Try GitLab patterns
        for pattern in gitlab_patterns:
            match = re.match(pattern, url)
            if match:
                owner, name = match.groups()
                return 'gitlab', owner, name

        # If we get here, no pattern matched
        raise ValueError(f"Could not parse repository URL: {url}")

    def fetch_repo(self, url: str, local_path: Optional[str] = None, persistent: bool = False) -> str:
        """
        Clone repository to a directory or use an existing local copy.

        Args:
            url: Repository URL (GitHub or GitLab) or local path
            local_path: Path to existing local repository (optional)
            persistent: If True, clone to a persistent directory instead of temporary

        Returns:
            Path to the repository

        Raises:
            GitCommandError: If cloning fails
            ValueError: If local path doesn't exist or isn't a git repository
        """
        self.repo_url = url

        # Check if we're using a local repository path
        if local_path:
            if not os.path.exists(local_path):
                raise ValueError(f"Local repository path does not exist: {local_path}")

            try:
                # Verify it's a git repository
                repo = Repo(local_path)
                # Get repository name from directory name
                self.repo_name = os.path.basename(os.path.normpath(local_path))
                # Try to get remote information if available
                try:
                    origin = repo.remotes.origin.url
                    try:
                        self.repo_type, self.repo_owner, _ = self._parse_repo_url(origin)
                    except ValueError:
                        # If we can't parse the remote URL, use placeholders
                        self.repo_type = "local"
                        self.repo_owner = "local"
                except (AttributeError, ValueError):
                    # No remote or can't parse URL
                    self.repo_type = "local"
                    self.repo_owner = "local"

                logger.info(f"Using local repository: {local_path}")
                self.repo_path = local_path
                return local_path

            except (GitCommandError, InvalidGitRepositoryError) as e:
                raise ValueError(f"Not a valid git repository: {local_path}")

        # Parse repository URL for remote repositories
        try:
            self.repo_type, self.repo_owner, self.repo_name = self._parse_repo_url(url)
            logger.info(f"Parsed repository: {self.repo_type}/{self.repo_owner}/{self.repo_name}")
        except ValueError as e:
            logger.error(f"Error parsing URL: {e}")
            raise

        # Determine target directory
        if persistent:
            # Create a persistent directory in the current working directory
            target_dir = os.path.join(os.getcwd(), "repos")
            os.makedirs(target_dir, exist_ok=True)
            self.repo_path = os.path.join(target_dir, f"{self.repo_owner}_{self.repo_name}")
            # Don't set temp_dir since we don't want to clean it up
        else:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="repo_analysis_")
            self.repo_path = os.path.join(self.temp_dir, self.repo_name)

        # If persistent and directory already exists, just return it
        if persistent and os.path.exists(self.repo_path):
            logger.info(f"Using existing repository at {self.repo_path}")
            return self.repo_path

        # Construct authentication URL if tokens are provided
        clone_url = url
        if self.repo_type == 'github' and self.github_token:
            # Format: https://{token}@github.com/{owner}/{repo}.git
            clone_url = f"https://{self.github_token}@github.com/{self.repo_owner}/{self.repo_name}.git"
        elif self.repo_type == 'gitlab' and self.gitlab_token:
            # Format: https://oauth2:{token}@gitlab.com/{owner}/{repo}.git
            clone_url = f"https://oauth2:{self.gitlab_token}@gitlab.com/{self.repo_owner}/{self.repo_name}.git"

        # Clone repository
        try:
            logger.info(f"Cloning repository to {self.repo_path}")
            Repo.clone_from(clone_url, self.repo_path)
            logger.info(f"Repository cloned successfully")
            return self.repo_path
        except GitCommandError as e:
            logger.error(f"Error cloning repository: {e}")
            # Only clean up if using temporary directory
            if not persistent:
                self.cleanup()
            raise

    def get_basic_repo_info(self) -> Dict:
        """
        Get basic information about the repository.

        Returns:
            Dictionary containing repository metadata

        Raises:
            ValueError: If repository has not been fetched yet
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            raise ValueError("Repository not fetched. Call fetch_repo() first.")

        try:
            repo = Repo(self.repo_path)

            # Get total number of commits
            commit_count = sum(1 for _ in repo.iter_commits())

            # Get list of branches
            branches = [b.name for b in repo.branches]

            # Get number of contributors
            contributors = {}
            for commit in repo.iter_commits():
                author = f"{commit.author.name} <{commit.author.email}>"
                if author in contributors:
                    contributors[author] += 1
                else:
                    contributors[author] = 1

            # Get last commit date
            last_commit_date = next(repo.iter_commits()).committed_datetime.isoformat()

            # Get languages used (simple implementation based on file extensions)
            languages = self.detect_languages()

            # Get README content if available
            readme_content = self.get_readme_content()

            # Get GitHub-specific files and metadata
            github_files = self.get_github_files()

            return {
                "name": self.repo_name,
                "owner": self.repo_owner,
                "type": self.repo_type,
                "url": self.repo_url,
                "commit_count": commit_count,
                "branches": branches,
                "default_branch": repo.active_branch.name,
                "contributors": contributors,
                "contributor_count": len(contributors),
                "last_commit_date": last_commit_date,
                "languages": languages,
                "has_readme": readme_content is not None,
                "github_files": github_files
            }
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            raise

    def detect_languages(self) -> Dict[str, int]:
        """
        Detect programming languages used in the repository based on file extensions.

        Returns:
            Dictionary mapping language names to line counts
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            raise ValueError("Repository not fetched. Call fetch_repo() first.")

        # Extension to language mapping
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React/JSX',
            '.tsx': 'React/TypeScript',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.h': 'C/C++ Header',
            '.cs': 'C#',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.rs': 'Rust',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sql': 'SQL',
            '.sh': 'Shell',
            '.bat': 'Batch',
            '.ps1': 'PowerShell',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.r': 'R',
            '.dart': 'Dart',
            '.ex': 'Elixir',
            '.exs': 'Elixir',
        }

        # Initialize counters
        language_lines = {}

        # Directories to exclude
        exclude_dirs = {'.git', 'node_modules', 'venv', 'env', '.env', 'build', 'dist', 'target'}

        # Walk through repository files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)

                if ext in ext_to_lang:
                    lang = ext_to_lang[ext]
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            line_count = sum(1 for _ in f)

                        if lang in language_lines:
                            language_lines[lang] += line_count
                        else:
                            language_lines[lang] = line_count
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")

        return language_lines

    def get_readme_content(self) -> Optional[str]:
        """
        Get content of repository README file, if available.

        Returns:
            README content as string, or None if not found
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            raise ValueError("Repository not fetched. Call fetch_repo() first.")

        # Check for common README filenames
        readme_patterns = [
            'README.md',
            'README.txt',
            'README',
            'Readme.md',
            'readme.md'
        ]

        for pattern in readme_patterns:
            readme_path = os.path.join(self.repo_path, pattern)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Could not read README file {readme_path}: {e}")

        return None

    def get_file_list(self, max_files: int = 1000) -> List[str]:
        """
        Get list of files in the repository.

        Args:
            max_files: Maximum number of files to return

        Returns:
            List of file paths relative to repository root
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            raise ValueError("Repository not fetched. Call fetch_repo() first.")

        # Directories to exclude
        exclude_dirs = {'.git', 'node_modules', 'venv', 'env', '.env', 'build', 'dist', 'target'}

        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            # Get relative path
            rel_root = os.path.relpath(root, self.repo_path)
            if rel_root == '.':
                rel_root = ''

            for file in filenames:
                if len(files) >= max_files:
                    return files

                rel_path = os.path.join(rel_root, file)
                files.append(rel_path)

        return files

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get content of a specific file in the repository.

        Args:
            file_path: Path to file, relative to repository root

        Returns:
            File content as string, or None if file cannot be read
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            raise ValueError("Repository not fetched. Call fetch_repo() first.")

        abs_path = os.path.join(self.repo_path, file_path)

        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read file {abs_path}: {e}")
            return None

    def get_github_files(self) -> Dict[str, Optional[str]]:
        """
        Get GitHub-specific files and metadata.

        Returns:
            Dictionary containing GitHub files and their content
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            raise ValueError("Repository not fetched. Call fetch_repo() first.")

        github_files = {}

        # List of common GitHub files to check
        files_to_check = [
            # GitHub configuration files
            '.github/workflows',       # GitHub Actions workflows directory
            '.github/ISSUE_TEMPLATE',  # Issue templates directory
            '.github/FUNDING.yml',     # Funding information
            '.github/dependabot.yml',  # Dependabot configuration
            'CODEOWNERS',              # Code owners definition
            '.github/PULL_REQUEST_TEMPLATE.md',  # PR template

            # GitHub metadata files
            'CONTRIBUTING.md',         # Contribution guidelines
            'CODE_OF_CONDUCT.md',      # Code of conduct
            'SECURITY.md',             # Security policy
            'SUPPORT.md',              # Support information
            '.github/ISSUE_TEMPLATE/bug_report.md',
            '.github/ISSUE_TEMPLATE/feature_request.md',

            # CI/CD and configuration files
            '.github/workflows/ci.yml',
            '.github/workflows/cd.yml',
            '.github/workflows/release.yml',
            '.github/dependabot.yml',

            # GitHub pages
            '.github/pages',

            # GitHub specific config
            '.github/config.yml',
            '.github/settings.yml'
        ]

        for file_path in files_to_check:
            full_path = os.path.join(self.repo_path, file_path)

            # Check if it's a directory
            if os.path.isdir(full_path):
                # Get list of files in the directory
                github_files[file_path] = []
                for root, _, files in os.walk(full_path):
                    rel_path = os.path.relpath(root, self.repo_path)
                    for file in files:
                        file_rel_path = os.path.join(rel_path, file)
                        content = self.get_file_content(file_rel_path)
                        if content:
                            github_files[file_path].append({
                                'path': file_rel_path,
                                'content': content
                            })
            # Check if it's a file
            elif os.path.isfile(full_path):
                content = self.get_file_content(file_path)
                if content:
                    github_files[file_path] = content

        return github_files


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clone and analyze GitHub/GitLab repositories")
    parser.add_argument("--url", required=True, help="Repository URL")
    parser.add_argument("--github-token", help="GitHub personal access token")
    parser.add_argument("--gitlab-token", help="GitLab personal access token")

    args = parser.parse_args()

    try:
        with RepoFetcher(github_token=args.github_token, gitlab_token=args.gitlab_token) as fetcher:
            repo_path = fetcher.fetch_repo(args.url)
            print(f"Repository cloned to: {repo_path}")

            info = fetcher.get_basic_repo_info()
            print("\nRepository Information:")
            for key, value in info.items():
                if key != "contributors":  # Skip printing all contributors
                    print(f"  {key}: {value}")

            print(f"\nTop 10 files:")
            for file in fetcher.get_file_list(max_files=10):
                print(f"  {file}")

            readme = fetcher.get_readme_content()
            if readme:
                print(f"\nREADME Preview (first 500 chars):")
                print(readme[:500] + ("..." if len(readme) > 500 else ""))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)