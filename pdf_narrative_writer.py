from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os
import time


class PDFNarrativeWriter:
    def __init__(self, output_dir: str = "output_reports"):
        """
        Initializes the PDF writer.

        Args:
            output_dir: Directory where PDF files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_pdf(self, text: str, repo_name: str, role: str, key_findings: list[str] = None) -> str:
        filename = f"{repo_name}_{role}_report_{int(time.time())}.pdf"
        output_path = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        normal_style.fontName = 'Helvetica'
        normal_style.fontSize = 11
        normal_style.leading = 15

        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=18,
            leading=24,
            alignment=1,
            spaceAfter=24
        )

        flowables = [Paragraph(f"Repository Analysis Report: {repo_name} ({role.title()} Perspective)", title_style)]

        # Add key findings if present
        if key_findings:
            heading = Paragraph("Key Findings", styles["Heading2"])
            flowables.append(heading)
            flowables.append(Spacer(1, 12))

            for point in key_findings:
                bullet = f"• {point.strip()}"
                flowables.append(Paragraph(bullet, normal_style))
                flowables.append(Spacer(1, 6))

            flowables.append(Spacer(1, 18))

        # Main narrative
        for para in text.split('\n'):
            para = para.strip()
            if para:
                flowables.append(Paragraph(para, normal_style))
                flowables.append(Spacer(1, 12))

        doc.build(flowables)
        return output_path

    def write_pdf_from_file(self, file_path: str, repo_name: str, role: str) -> str:
        """
        Read a stitched report from a .txt file and generate a PDF.

        Args:
            file_path: Path to the narrative .txt file.
            repo_name: Name of the repository.
            role: Role for whom the report is written.

        Returns:
            Path to the generated PDF.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            narrative_text = f.read()

        #return self.write_pdf(narrative_text, repo_name, role)
        return self.write_pdf(
        text=narrative_text,
        repo_name="fastapi-users",
        role="programmer",
        key_findings=key_findings
        )

key_findings = [
    "Python is the primary language with 88% code coverage.",
    "Modular architecture with plug-and-play authentication.",
    "Uses pytest with extensive fixtures and mocks for testing.",
    "Hatch is used for build and environment management.",
    "Well-documented with migration guides and structured configuration.",
    "No critical issues reported; focus on continuous improvement."
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a PDF report from a text file.")
    parser.add_argument("text_file", help="Path to the narrative .txt file")
    parser.add_argument("--repo", default="fastapi-users", help="Repository name")
    parser.add_argument("--role", default="programmer", help="Role for the report")
    parser.add_argument("--output_dir", default="output_reports", help="Directory to save the PDF")

    args = parser.parse_args()

    writer = PDFNarrativeWriter(output_dir=args.output_dir)

    pdf_path = writer.write_pdf_from_file(
        file_path=args.text_file,
        repo_name=args.repo,
        role=args.role
    )

    print(f"✅ PDF saved to: {pdf_path}")
