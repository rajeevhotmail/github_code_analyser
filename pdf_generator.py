#!/usr/bin/env python3
"""
PDF Generator Module

This module handles the generation of PDF reports from repository analysis results.
It provides functionality to:
1. Format repository information
2. Create structured question-answer sections
3. Generate professional PDF reports with full Unicode support
4. Include visual elements like charts and formatting

It implements comprehensive logging for tracking the PDF generation process.
"""

import os
import json
import time
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for matplotlib

# Setup module logger
logger = logging.getLogger("pdf_generator")
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class PDFGenerator:
    """
    Generates PDF reports from repository analysis results.
    Uses ReportLab for full Unicode support.
    """

    # Define role descriptions for the report
    ROLE_DESCRIPTIONS = {
        "programmer": (
            "This report provides a technical analysis of the repository from a programmer's perspective. "
            "It focuses on code structure, architecture, technologies used, and development practices."
        ),
        "ceo": (
            "This report provides a high-level analysis of the repository from a CEO's perspective. "
            "It focuses on business value, market positioning, resource requirements, and strategic considerations."
        ),
        "sales_manager": (
            "This report provides a product-focused analysis from a Sales Manager's perspective. "
            "It focuses on features, benefits, target customers, competitive positioning, and sales enablement information."
        )
    }

    # Define templates for different question types
    QUESTION_TEMPLATES = {
        "project_purpose": {
            "context": "Analyze the main purpose and core components of the repository: {repo_name}",
            "display_format": "descriptive",  # Uses paragraphs with headings
            "prompt_enhancement": "Focus on README.md, documentation files, and module docstrings to identify the primary purpose.",
            "section_title": "Project Purpose & Overview"
        },
        "dependencies": {
            "context": "Identify all dependencies for the repository: {repo_name}",
            "display_format": "table",  # Will render as a table in the PDF
            "prompt_enhancement": "Extract dependencies from package configuration files like requirements.txt, setup.py, or pyproject.toml.",
            "section_title": "Project Dependencies"
        },
        "build_process": {
            "context": "Describe the build and deployment process for: {repo_name}",
            "display_format": "steps",  # Will render as numbered steps
            "prompt_enhancement": "Look for CI configuration files, Dockerfiles, and deployment scripts.",
            "section_title": "Build & Deployment Process"
        },
        "api_usage": {
            "context": "Provide concrete examples of how to use the main API of: {repo_name}",
            "display_format": "code_sample",  # Will highlight code examples
            "prompt_enhancement": "Find examples in test files or documentation showing typical usage patterns.",
            "section_title": "API Usage Examples"
        },
        "architecture": {
            "context": "Describe the architecture and structure of: {repo_name}",
            "display_format": "descriptive",
            "prompt_enhancement": "Focus on how components interact, the design patterns used, and the overall system organization.",
            "section_title": "Architecture & Structure"
        },
        "testing": {
            "context": "Explain the testing approach used in: {repo_name}",
            "display_format": "descriptive",
            "prompt_enhancement": "Identify test frameworks, coverage tools, and testing patterns used in the project.",
            "section_title": "Testing Approach"
        },
        "default": {
            "context": "Analyze the following aspect of the repository: {question}",
            "display_format": "descriptive",
            "prompt_enhancement": "",
            "section_title": "Analysis"
        }
    }

    # Question type mapping (simple keyword matching)
    QUESTION_TYPE_MAPPING = {
        "what is the project about": "project_purpose",
        "what are the main components": "project_purpose",
        "what dependencies": "dependencies",
        "what is the build": "build_process",
        "what is the deployment": "build_process",
        "build/deployment process": "build_process",
        "give an example of using": "api_usage",
        "how to use": "api_usage",
        "architecture": "architecture",
        "structure": "architecture",
        "testing": "testing",
        "test framework": "testing",
    }

    def __init__(self, output_dir: str, log_level: int = logging.INFO):
        """
        Initialize the PDF generator.

        Args:
            output_dir: Directory to save generated PDFs
            log_level: Logging level for this generator instance
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(f"pdf_generator.{os.path.basename(output_dir)}")
        self.logger.setLevel(log_level)

        # Create file handler for this instance
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"pdf_{int(time.time())}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Initialized PDF generator with output directory: {output_dir}")

        # Check if ReportLab is installed
        self._has_reportlab = importlib.util.find_spec("reportlab") is not None
        if not self._has_reportlab:
            self.logger.warning(
                "reportlab package not found. Install with: pip install reportlab"
            )

        # Check if matplotlib is installed for charts
        self._has_matplotlib = importlib.util.find_spec("matplotlib") is not None
        if not self._has_matplotlib:
            self.logger.warning(
                "matplotlib package not found. Charts will not be available. "
                "Install with: pip install matplotlib"
            )

        # Repository info that will be set during generation
        self.repo_name = None
        self.repo_owner = None
        self.role = None

    def _detect_question_type(self, question: str) -> str:
        """
        Detect the question type based on keywords.

        Args:
            question: The question text

        Returns:
            The detected question type
        """
        question_lower = question.lower()

        for keyword, q_type in self.QUESTION_TYPE_MAPPING.items():
            if keyword in question_lower:
                self.logger.debug(f"Detected question type '{q_type}' for question: {question}")
                return q_type

        self.logger.debug(f"Using default question type for question: {question}")
        return "default"

    def _create_language_chart(self, repo_info, filename):
        """
        Create a chart of programming languages used in the repository.

        Args:
            repo_info: Repository information dictionary
            filename: Output filename for the chart

        Returns:
            Path to the chart image or None if failed
        """
        if not self._has_matplotlib:
            self.logger.warning("Cannot create chart: matplotlib not installed")
            return None

        # Check for languages data
        if 'languages' not in repo_info or not repo_info['languages']:
            self.logger.warning("No language data available for chart")
            return None

        languages = repo_info['languages']
        if not languages:
            self.logger.warning("Empty language data for chart")
            return None

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Sort languages by value
            sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)

            # Limit to top 8 languages
            top_langs = sorted_langs[:8]

            # Calculate total for percentages
            total = sum(count for _, count in top_langs)

            # Create pie chart
            labels = [lang for lang, _ in top_langs]
            sizes = [count for _, count in top_langs]
            percentages = [(count / total) * 100 for count in sizes]

            # Add percentage to labels
            labels = [f"{lang} ({pct:.1f}%)" for lang, pct in zip(labels, percentages)]

            # Use a better color palette
            colors = plt.cm.tab10.colors

            # Create figure with a professional look
            plt.figure(figsize=(8, 6), facecolor='white')
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=140, shadow=False, textprops={'fontsize': 9})
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Programming Languages Used', fontsize=14, fontweight='bold')

            # Add a legend outside the pie
            plt.legend(labels, loc="best", fontsize=8)

            # Save chart with higher DPI for better quality
            chart_path = os.path.join(self.output_dir, filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Created language chart: {chart_path}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error creating language chart: {e}", exc_info=True)
            return None

    def _escape_xml(self, text: str) -> str:
        """
        Escape XML special characters in text to prevent ReportLab parsing errors.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        if not text:
            return ""

        # Replace XML special characters
        replacements = [
            ('&', '&amp;'),  # Must be first to avoid double-escaping
            ('<', '&lt;'),
            ('>', '&gt;'),
            ('"', '&quot;'),
            ("'", '&#39;')
        ]

        for old, new in replacements:
            text = text.replace(old, new)

        return text

    def _safe_paragraph(self, text, style):
        """
        Safely create a paragraph, escaping XML and handling any errors.

        Args:
            text: Text content for the paragraph
            style: Paragraph style

        Returns:
            Paragraph object or a simple Spacer if paragraph creation fails
        """
        try:
            from reportlab.platypus import Paragraph, Spacer
            from reportlab.lib.units import inch

            escaped_text = self._escape_xml(text)
            return Paragraph(escaped_text, style)
        except Exception as e:
            self.logger.warning(f"Error creating paragraph: {e}. Text: {text[:50]}...")
            # Return a spacer instead of failing completely
            return Spacer(1, 0.1*inch)

    def _add_header_footer(self, canvas, doc):
        """
        Add header and footer to each page.

        Args:
            canvas: ReportLab canvas
            doc: ReportLab document
        """
        from reportlab.lib.units import inch

        canvas.saveState()

        # Header
        canvas.setFont('Helvetica', 8)
        if self.repo_name:
            canvas.drawString(72, doc.height + 50, f"Repository Analysis: {self.repo_name}")

        # Footer
        canvas.setFont('Helvetica', 8)
        page_num = canvas.getPageNumber()
        canvas.drawString(72, 40, f"Page {page_num}")
        canvas.drawString(400, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d')}")

        canvas.restoreState()

    def _create_cover_page(self, story, styles, repo_info):
        """
        Create an attractive cover page.

        Args:
            story: ReportLab story list
            styles: ReportLab styles
            repo_info: Repository information
        """
        from reportlab.platypus import Spacer, TableStyle, Table, Image, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        # Create title style
        title_style = styles['Title']
        subtitle_style = styles['Heading1']

        # Content for the cover
        content = []
        content.append(Spacer(1, 1*inch))
        content.append(self._safe_paragraph("Repository Analysis Report", title_style))
        content.append(Spacer(1, 0.5*inch))
        content.append(self._safe_paragraph(f"{repo_info['name']}", subtitle_style))
        content.append(Spacer(1, 0.5*inch))
        content.append(self._safe_paragraph(f"{self.role.title()} Perspective", subtitle_style))
        content.append(Spacer(1, 1*inch))

        today = datetime.now().strftime("%B %d, %Y")
        content.append(self._safe_paragraph(f"Generated on {today}", styles['Normal']))

        # If we have a language chart, add it to the cover
        if hasattr(self, 'language_chart_path') and os.path.exists(self.language_chart_path):
            content.append(Spacer(1, 0.5*inch))
            img = Image(self.language_chart_path, width=400, height=300)
            content.append(img)

        # Add the content with a border and background
        cover_table = Table([[c] for c in content], colWidths=[400])
        cover_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 1, colors.darkblue),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))

        story.append(cover_table)
        story.append(PageBreak())

    def _add_formatted_answer(self, story, qa_pair, styles):
        """
        Add the answer with appropriate formatting based on the question type.

        Args:
            story: ReportLab story list
            qa_pair: Question-answer pair dictionary
            styles: ReportLab styles dictionary
        """
        from reportlab.platypus import Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch

        question = qa_pair['question']
        answer = qa_pair['answer']

        # Detect question type
        question_type = self._detect_question_type(question)

        # Get display format from template or default to descriptive
        template = self.QUESTION_TEMPLATES.get(question_type, self.QUESTION_TEMPLATES['default'])
        display_format = template.get('display_format', 'descriptive')

        self.logger.debug(f"Using display format '{display_format}' for question type '{question_type}'")

        # Create custom styles for code blocks
        code_style = ParagraphStyle(
            'Code',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=5,
            borderRadius=2
        )

        # Table display format
        if display_format == 'table' and ':' in answer:
            # Parse the answer as a table if it contains key-value pairs
            rows = []
            headers = ["Item", "Description"]
            rows.append(headers)

            for line in answer.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    rows.append([key.strip(), value.strip()])

            if len(rows) > 1:  # Only create table if we have data rows
                # Create table with better styling
                table = Table(rows, colWidths=[150, 300])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ]))
                story.append(table)
                story.append(Spacer(1, 0.1 * inch))

                # Still add the text below for context
                story.append(self._safe_paragraph(answer, styles['Normal']))
                return

        # Steps display format
        elif display_format == 'steps':
            # Look for numbered lists or create them
            if answer.strip().startswith('1.') or answer.strip().startswith('1)'):
                # Already formatted as steps, use as is
                story.append(self._safe_paragraph(answer, styles['Normal']))
                return
            else:
                # Format as numbered steps
                steps = [s for s in answer.split('\n') if s.strip()]
                for i, step in enumerate(steps, 1):
                    if step.strip():
                        story.append(self._safe_paragraph(f"{i}. {step}", styles['Normal']))
                        story.append(Spacer(1, 0.05 * inch))
                return

        # Code sample display format
        elif display_format == 'code_sample':
            # Look for code blocks with ``` or parse out code sections
            if '```' in answer:
                parts = answer.split('```')
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Regular text
                        if part.strip():
                            story.append(self._safe_paragraph(part, styles['Normal']))
                    else:  # Code block
                        # Extract language if specified (```python)
                        code_text = part
                        if '\n' in part:
                            first_line, rest = part.split('\n', 1)
                            if first_line.strip() and not first_line.strip().startswith('#'):
                                # This is likely a language specifier
                                code_text = rest

                        story.append(self._safe_paragraph(code_text, code_style))
                        story.append(Spacer(1, 0.1 * inch))
                return
            elif 'example:' in answer.lower() or 'examples:' in answer.lower():
                # Try to find code examples by splitting on 'Example:'
                for part in re.split(r'(?i)example(s?):', answer):
                    if 'import' in part or 'class' in part or 'def ' in part or '=' in part:
                        # This part likely contains code
                        story.append(self._safe_paragraph(part, code_style))
                    else:
                        if part.strip():
                            story.append(self._safe_paragraph(part, styles['Normal']))
                return

        # Default: descriptive format
        story.append(self._safe_paragraph(answer, styles['Normal']))

    def generate_pdf(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a PDF report from repository analysis results.
        Uses ReportLab for Unicode support.

        Args:
            report_data: Dictionary with repository analysis results

        Returns:
            Path to the generated PDF file
        """
        if not self._has_reportlab:
            self.logger.error("Cannot generate PDF: reportlab not installed")
            return ""

        try:
            # Import ReportLab components
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.platypus import PageBreak, Flowable
            from reportlab.lib.units import inch, cm
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

            # Import for TOC
            from reportlab.platypus.tableofcontents import TableOfContents

            # Create a bookmark flowable
            class Bookmark(Flowable):
                def __init__(self, title, key):
                    self.title = title
                    self.key = key
                    self.width = 0
                    self.height = 0

                def draw(self):
                    self.canv.bookmarkPage(self.key)
                    self.canv.addOutlineEntry(self.title, self.key, 0, 0)

            # Extract data
            repository = report_data['repository']
            role = report_data['role']
            qa_pairs = report_data['qa_pairs']

            # Set instance variables for header/footer
            self.repo_name = repository.get('name', 'Unknown')
            self.repo_owner = repository.get('owner', 'Unknown')
            self.role = role

            # Prepare output file path
            output_file = os.path.join(
                self.output_dir,
                f"{self.repo_name}_{role}_report_{int(time.time())}.pdf"
            )

            # Create the document
            doc = SimpleDocTemplate(
                output_file,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Get styles
            styles = getSampleStyleSheet()

            # Create custom styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontSize=24,
                alignment=TA_CENTER,
                spaceAfter=20
            )

            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Title'],
                fontSize=18,
                alignment=TA_CENTER,
                spaceAfter=12
            )

            heading1_style = ParagraphStyle(
                'Heading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=10,
                keepWithNext=True
            )

            heading2_style = ParagraphStyle(
                'Heading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=8,
                keepWithNext=True
            )

            heading3_style = ParagraphStyle(
                'Heading3',
                parent=styles['Heading3'],
                fontSize=12,
                spaceAfter=6,
                keepWithNext=True
            )

            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                leading=14,
                spaceAfter=6
            )

            # Better body text style
            body_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                leading=14,
                spaceAfter=6
            )

            # Caption style for charts and tables
            caption_style = ParagraphStyle(
                'Caption',
                parent=styles['Italic'],
                fontSize=9,
                alignment=TA_CENTER,
                textColor=colors.darkgrey
            )

            # Story holds all elements
            story = []

            # Generate language chart
            chart_file = f"{self.repo_name}_languages.png"
            self.language_chart_path = self._create_language_chart(repository, chart_file)

            # Create cover page
            self._create_cover_page(story, styles, repository)

            # Table of Contents
            story.append(self._safe_paragraph("Table of Contents", heading1_style))
            story.append(Bookmark("Table of Contents", "toc"))

            # Create Table of Contents object
            toc = TableOfContents()
            toc.levelStyles = [
                ParagraphStyle(name='TOCHeading1', fontSize=12, leading=16, leftIndent=20, firstLineIndent=-20),
                ParagraphStyle(name='TOCHeading2', fontSize=10, leading=14, leftIndent=40, firstLineIndent=-20),
            ]
            story.append(toc)
            story.append(PageBreak())

            # Executive Summary
            story.append(Bookmark("Executive Summary", "exec-summary"))
            story.append(self._safe_paragraph("Executive Summary", heading1_style))

            # Add role-specific executive summary
            role_description = self.ROLE_DESCRIPTIONS.get(role, "This report provides an analysis of the repository.")
            story.append(self._safe_paragraph(role_description, body_style))
            story.append(Spacer(1, 0.1*inch))

            # Add repository highlights
            story.append(self._safe_paragraph("Repository Highlights:", heading3_style))

            # Extract highlights from information
            highlights = [
                f"Repository: {repository.get('name', 'Unknown')} by {repository.get('owner', 'Unknown')}",
                f"Contains {repository.get('commit_count', 'Unknown')} commits from {repository.get('contributor_count', 'Unknown')} contributors"
            ]

            # Add language information if available
            if 'languages' in repository and repository['languages']:
                primary_lang = sorted(repository['languages'].items(), key=lambda x: x[1], reverse=True)[0][0]
                highlights.append(f"Primary language: {primary_lang}")

            # Add the highlights as a bulleted list
            for highlight in highlights:
                story.append(self._safe_paragraph(f"• {highlight}", body_style))

            story.append(Spacer(1, 0.2*inch))
            story.append(self._safe_paragraph(
                "The following report contains detailed answers to key questions from a "
                f"{role.replace('_', ' ')} perspective, based on automated analysis of the "
                "repository content.", body_style))

            story.append(PageBreak())

            # Repository Information
            story.append(Bookmark("Repository Information", "repo-info"))
            story.append(self._safe_paragraph("Repository Information", heading1_style))

            # Create a table for repository info with better styling
            repo_data = [
                ["Name:", repository.get('name', 'Unknown')],
                ["Owner:", repository.get('owner', 'Unknown')],
                ["URL:", repository.get('url', 'Unknown')]
            ]

            if 'languages' in repository and repository['languages']:
                lang_str = ", ".join(repository['languages'].keys())
                repo_data.append(["Languages:", lang_str])

            repo_data.append(["Commit Count:", str(repository.get('commit_count', 'Unknown'))])
            repo_data.append(["Contributors:", str(repository.get('contributor_count', 'Unknown'))])

            repo_table = Table(repo_data, colWidths=[100, 350])
            repo_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))

            story.append(repo_table)
            story.append(Spacer(1, 0.5*inch))

            # Add language chart if available
            if self.language_chart_path and os.path.exists(self.language_chart_path):
                story.append(self._safe_paragraph("Programming Languages Distribution:", heading2_style))
                story.append(Spacer(1, 0.2*inch))

                # Add the chart
                img = Image(self.language_chart_path, width=400, height=300)
                story.append(img)

                # Add caption
                story.append(self._safe_paragraph(
                    "Figure 1: Distribution of programming languages in the repository",
                    caption_style))

            # Introduction
            story.append(PageBreak())
            story.append(Bookmark("Introduction", "intro"))
            story.append(self._safe_paragraph("Introduction", heading1_style))

            story.append(self._safe_paragraph(role_description, body_style))

            story.append(Spacer(1, 0.3*inch))
            story.append(self._safe_paragraph(
                "The following pages contain answers to key questions relevant to this perspective, "
                "based on automated analysis of the repository content.", body_style))

            # Q&A sections with better styling
            for i, qa_pair in enumerate(qa_pairs, 1):
                story.append(PageBreak())

                question = qa_pair['question']

                # Detect question type and get section title
                question_type = self._detect_question_type(question)
                template = self.QUESTION_TEMPLATES.get(question_type, self.QUESTION_TEMPLATES['default'])
                section_title = template.get('section_title', "Analysis")

                # Create section header and bookmark
                bookmark_id = f"question-{i}"
                story.append(Bookmark(f"Question {i}: {question}", bookmark_id))
                story.append(self._safe_paragraph(f"Question {i}: {question}", heading1_style))

                # Add a divider line
                story.append(Spacer(1, 0.1*inch))

                # Add formatted answer
                self._add_formatted_answer(story, qa_pair, styles)

                # Add a "Key Findings" section
                if len(qa_pair['answer']) > 200:  # Only add for substantial answers
                    story.append(Spacer(1, 0.2*inch))
                    story.append(self._safe_paragraph("Key Findings:", heading3_style))

                    # Extract key points (simple heuristic - look for sentences with important words)
                    key_points = []

                    # Keywords that suggest important information
                    importance_markers = [
                        "main", "primary", "important", "key", "critical", "essential",
                        "significant", "notable", "major", "central", "core"
                    ]

                    # Simple extraction of important sentences
                    sentences = qa_pair['answer'].split('. ')
                    for sentence in sentences:
                        if any(marker in sentence.lower() for marker in importance_markers):
                            if sentence and sentence.strip():
                                # Clean up the sentence
                                clean_sentence = sentence.strip()
                                if not clean_sentence.endswith('.'):
                                    clean_sentence += '.'
                                key_points.append(clean_sentence)

                    # If no key points found with markers, take first 2-3 sentences
                    if not key_points and len(sentences) > 3:
                        key_points = [s.strip() + '.' for s in sentences[:3] if s.strip()]

                    # Add key points as bullets
                    for point in key_points[:3]:  # Limit to top 3 points
                        story.append(self._safe_paragraph(f"• {point}", body_style))

                # Add sources if available
                if 'supporting_chunks' in qa_pair and qa_pair['supporting_chunks']:
                    story.append(Spacer(1, 0.3*inch))
                    story.append(self._safe_paragraph("Based on information from:", heading3_style))

                    for j, chunk in enumerate(qa_pair['supporting_chunks'][:3], 1):  # Limit to top 3 sources
                        source = f"{j}. {chunk['file_path']}"
                        if chunk.get('name'):
                            source += f" ({chunk['name']})"
                        story.append(self._safe_paragraph(source, styles['Italic']))

            # Conclusion
            story.append(PageBreak())
            story.append(Bookmark("Conclusion", "conclusion"))
            story.append(self._safe_paragraph("Conclusion", heading1_style))

            # Add a summary based on role
            if role == "programmer":
                conclusion_text = (
                    f"This technical analysis of {self.repo_name} covered the core architecture, "
                    "dependencies, and development practices. The repository demonstrates "
                    "a structured approach to software development with clear patterns and practices. "
                    "For more detailed information on specific implementation details, refer to the "
                    "repository itself and its documentation."
                )
            elif role == "ceo":
                conclusion_text = (
                    f"This executive analysis of {self.repo_name} examined the business value, "
                    "market positioning, and strategic considerations for this project. "
                    "The analysis highlights key opportunity areas and potential challenges. "
                    "For a deeper strategic assessment, consider consulting with technical "
                    "stakeholders who are familiar with the project."
                )
            elif role == "sales_manager":
                conclusion_text = (
                    f"This product-focused analysis of {self.repo_name} identified key features, "
                    "benefits, and competitive positioning. The information provided can "
                    "support sales enablement and customer outreach efforts. For customized "
                    "messaging for specific market segments, consider additional analysis in "
                    "collaboration with the product and engineering teams."
                )
            else:
                conclusion_text = (
                    "This report was generated automatically by analyzing the repository content. "
                    "The analysis is based on the code, documentation, and configuration files present in the repository. "
                    "For more detailed information, please refer to the repository itself or contact the development team."
                )

            story.append(self._safe_paragraph(conclusion_text, body_style))

            # Add recommendations section
            story.append(Spacer(1, 0.3*inch))
            story.append(self._safe_paragraph("Recommendations", heading2_style))

            # Generic recommendations based on role
            recommendations = []
            if role == "programmer":
                recommendations = [
                    "Review the codebase with the development team to identify optimization opportunities",
                    "Consider running additional static analysis tools for deeper code quality insights",
                    "Compare the identified architecture with system documentation to ensure alignment"
                ]
            elif role == "ceo":
                recommendations = [
                    "Evaluate the project's alignment with current business objectives",
                    "Assess resource requirements against projected ROI",
                    "Consider competitive positioning based on the identified capabilities"
                ]
            elif role == "sales_manager":
                recommendations = [
                    "Create targeted messaging highlighting the key features identified",
                    "Develop use case scenarios based on the capabilities uncovered",
                    "Compare with competitor offerings to emphasize unique selling points"
                ]

            # Add recommendations as bullet points
            for rec in recommendations:
                story.append(self._safe_paragraph(f"• {rec}", body_style))

            # Build the PDF with header/footer
            doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)

            self.logger.info(f"Generated PDF report: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}", exc_info=True)
            return ""

# Import for regex support
import re

# Function to process a report data file
def process_report_file(report_file, output_dir, log_level=logging.INFO):
    """
    Process a single report data file and generate a PDF.

    Args:
        report_file: Path to the JSON report data file
        output_dir: Directory to save the PDF
        log_level: Logging level

    Returns:
        Path to the generated PDF or empty string if failed
    """
    try:
        # Load report data
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        # Initialize PDF generator
        generator = PDFGenerator(
            output_dir=output_dir,
            log_level=log_level
        )

        # Generate PDF
        pdf_file = generator.generate_pdf(report_data)

        return pdf_file

    except Exception as e:
        logger.error(f"Error processing report file {report_file}: {e}", exc_info=True)
        return ""

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF reports from repository analysis results")
    parser.add_argument("--report-data", help="JSON file with report data")
    parser.add_argument("--reports-dir", help="Directory containing report data files to batch process")
    parser.add_argument("--output-dir", default="./reports", help="Directory to save PDFs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Ensure we have a report source
    if not (args.report_data or args.reports_dir):
        print("Error: Either --report-data or --reports-dir must be specified")
        parser.print_help()
        exit(1)

    # Process a single file
    if args.report_data:
        pdf_file = process_report_file(args.report_data, args.output_dir, log_level)

        if pdf_file:
            print(f"PDF report generated: {pdf_file}")
        else:
            print("Failed to generate PDF report")

    # Process a directory of files
    elif args.reports_dir:
        import glob

        # Find all JSON files in the directory
        report_files = glob.glob(os.path.join(args.reports_dir, "*.json"))

        if not report_files:
            print(f"No JSON files found in {args.reports_dir}")
            exit(0)

        print(f"Found {len(report_files)} report files to process")

        # Process each file
        success_count = 0
        for i, report_file in enumerate(report_files, 1):
            print(f"Processing file {i}/{len(report_files)}: {os.path.basename(report_file)}")

            pdf_file = process_report_file(report_file, args.output_dir, log_level)

            if pdf_file:
                print(f"  Success: {os.path.basename(pdf_file)}")
                success_count += 1
            else:
                print(f"  Failed to process {os.path.basename(report_file)}")

        print(f"Processed {len(report_files)} files: {success_count} successful, {len(report_files) - success_count} failed")