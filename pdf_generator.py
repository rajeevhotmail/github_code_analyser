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
import sys
import time
import logging
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

        languages = repo_info.get('languages', {})
        if not languages:
            self.logger.warning("No language data available for chart")
            return None

        try:
            import matplotlib.pyplot as plt

            # Sort languages by line count
            sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)

            # Limit to top 10 languages
            top_langs = sorted_langs[:10]

            # Create pie chart
            labels = [lang for lang, _ in top_langs]
            sizes = [count for _, count in top_langs]

            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Programming Languages Used')

            # Save chart
            chart_path = os.path.join(self.output_dir, filename)
            plt.savefig(chart_path)
            plt.close()

            self.logger.info(f"Created language chart: {chart_path}")
            return chart_path

        except Exception as e:
            self.logger.error(f"Error creating language chart: {e}", exc_info=True)
            return None

    # Add this helper function to your PDFGenerator class
    # Add this helper function to your PDFGenerator class
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
            from reportlab.platypus import PageBreak
            from reportlab.lib.units import inch, cm
            from reportlab.lib.enums import TA_CENTER, TA_LEFT

            # Extract data
            repository = report_data['repository']
            role = report_data['role']
            qa_pairs = report_data['qa_pairs']

            repo_name = repository.get('name', 'Unknown')
            repo_owner = repository.get('owner', 'Unknown')
            repo_url = repository.get('url', 'Unknown')

            # Prepare output file path
            output_file = os.path.join(
                self.output_dir,
                f"{repo_name}_{role}_report_{int(time.time())}.pdf"
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
                fontSize=20,
                alignment=TA_CENTER,
                spaceAfter=20
            )

            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Title'],
                fontSize=16,
                alignment=TA_CENTER,
                spaceAfter=10
            )

            heading1_style = ParagraphStyle(
                'Heading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=10
            )

            heading2_style = ParagraphStyle(
                'Heading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=8
            )

            normal_style = styles['Normal']

            # Story holds all elements
            story = []

            # Cover Page
            story.append(self._safe_paragraph(f"Repository Analysis Report: {repo_name}", title_style))
            story.append(self._safe_paragraph(f"{role.capitalize()} Perspective", subtitle_style))
            story.append(Spacer(1, 0.5*inch))

            today = datetime.now().strftime("%B %d, %Y")
            story.append(self._safe_paragraph(f"Generated on {today}", styles['Normal']))

            story.append(Spacer(1, 1*inch))

            # Repository Information
            story.append(PageBreak())
            story.append(self._safe_paragraph("Repository Information", heading1_style))

            # Create a table for repository info
            repo_data = [
                ["Name:", repo_name],
                ["Owner:", repo_owner],
                ["URL:", repo_url]
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
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            story.append(repo_table)
            story.append(Spacer(1, 0.5*inch))

            # Add language chart if matplotlib is available
            if self._has_matplotlib and 'languages' in repository and repository['languages']:
                chart_file = f"{repo_name}_languages.png"
                chart_path = self._create_language_chart(repository, chart_file)

                if chart_path and os.path.exists(chart_path):
                    story.append(self._safe_paragraph("Programming Languages Distribution:", heading2_style))
                    story.append(Spacer(1, 0.2*inch))

                    # Add the chart
                    img = Image(chart_path, width=400, height=300)
                    story.append(img)

            # Introduction
            story.append(PageBreak())
            story.append(self._safe_paragraph("Introduction", heading1_style))

            role_description = self.ROLE_DESCRIPTIONS.get(role, "This report provides an analysis of the repository.")
            story.append(self._safe_paragraph(role_description, normal_style))

            story.append(Spacer(1, 0.3*inch))
            story.append(self._safe_paragraph(
                "The following pages contain answers to key questions relevant to this perspective, "
                "based on automated analysis of the repository content.", normal_style))

            # Q&A sections
            for i, qa_pair in enumerate(qa_pairs, 1):
                story.append(PageBreak())

                question = qa_pair['question']
                answer = qa_pair['answer']

                # Question Header
                story.append(self._safe_paragraph(f"Question {i}: {question}", heading1_style))

                # Answer
                story.append(self._safe_paragraph(answer, normal_style))

                # Add sources if available
                if 'supporting_chunks' in qa_pair and qa_pair['supporting_chunks']:
                    story.append(Spacer(1, 0.3*inch))
                    story.append(self._safe_paragraph("Based on information from:", heading2_style))

                    for j, chunk in enumerate(qa_pair['supporting_chunks'][:3], 1):  # Limit to top 3 sources
                        source = f"{j}. {chunk['file_path']}"
                        if chunk.get('name'):
                            source += f" ({chunk['name']})"
                        story.append(self._safe_paragraph(source, styles['Italic']))

            # Conclusion
            story.append(PageBreak())
            story.append(self._safe_paragraph("Conclusion", heading1_style))

            story.append(self._safe_paragraph(
                "This report was generated automatically by analyzing the repository content. "
                "The analysis is based on the code, documentation, and configuration files present in the repository. "
                "For more detailed information, please refer to the repository itself or contact the development team.",
                normal_style))

            # Build the PDF
            doc.build(story)

            self.logger.info(f"Generated PDF report: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}", exc_info=True)
            return ""
    def generate_ascii_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a plain ASCII text report from repository analysis results.

        Args:
            report_data: Dictionary with repository analysis results

        Returns:
            String containing the plain text report
        """
        try:
            # Extract data
            repository = report_data['repository']
            role = report_data['role']
            qa_pairs = report_data['qa_pairs']

            repo_name = repository.get('name', 'Unknown')
            repo_owner = repository.get('owner', 'Unknown')
            repo_url = repository.get('url', 'Unknown')

            report = []

            # Add header
            report.append(f"Repository Analysis Report: {repo_name}")
            report.append(f"Role: {role.capitalize()}")
            report.append(f"Generated on {datetime.now().strftime('%B %d, %Y')}")
            report.append("=" * 80)
            report.append("\nRepository Information:\n")
            report.append(f"- Name: {repo_name}")
            report.append(f"- Owner: {repo_owner}")
            report.append(f"- URL: {repo_url}")
            if 'languages' in repository and repository['languages']:
                lang_str = ", ".join(repository['languages'].keys())
                report.append(f"- Languages: {lang_str}")
            report.append(f"- Commit Count: {repository.get('commit_count', 'Unknown')}")
            report.append(f"- Contributors: {repository.get('contributor_count', 'Unknown')}")

            report.append("\nIntroduction:")
            report.append(
                "This report provides an analysis of the repository content, focusing on "
                "code structure, architecture, and technologies used."
            )

            # Add Q&A as paragraphs
            for qa_pair in qa_pairs:
                title = qa_pair['question']
                answer = qa_pair['answer']
                report.append("\n" + "=" * 80)
                report.append(f"\n{title}:")
                report.append(answer.replace("\n", " "))  # Simplify formatting

            report.append("\n" + "=" * 80)
            report.append("\nConclusion:")
            report.append(
                "This report was generated by analyzing repository content. For more information, "
                "refer to the repository or contact the development team."
            )

            # Convert list to plain text
            ascii_report = "\n".join(report)
            return ascii_report

        except Exception as e:
            print(f"Error generating ASCII report: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF reports from repository analysis results")
    parser.add_argument("--report-data", required=True, help="JSON file with report data")
    parser.add_argument("--output-dir", default="./reports", help="Directory to save PDFs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Load report data
    with open(args.report_data, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    # Initialize PDF generator
    generator = PDFGenerator(
        output_dir=args.output_dir,
        log_level=log_level
    )

    # Generate PDF
    pdf_file = generator.generate_pdf(report_data)

    if pdf_file:
        print(f"PDF report generated: {pdf_file}")
    else:
        print("Failed to generate PDF report")