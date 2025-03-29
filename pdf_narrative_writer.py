import os
import time
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from section_headers import ROLE_SECTION_HEADERS


def remove_img_tags(text):
    return re.sub(r'<img[^>]*>', '', text)


class TOCEntry(Paragraph):
    def __init__(self, text, style, level=0):
        super().__init__(text, style)
        self.level = level
        self.docref = None

    def draw(self):
        self.canv.bookmarkPage(self.getPlainText())
        if self.docref:
            self.docref.notify('TOCEntry', (self.level, self.getPlainText(), self.canv.getPageNumber()))
        super().draw()


class PDFNarrativeWriter:
    def __init__(self, output_dir: str = "output_reports"):
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
        normal = styles["Normal"]
        heading1 = styles["Heading1"]
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=18,
            leading=24,
            alignment=1,
            spaceAfter=24
        )

        flowables = []

        # Title
        flowables.append(Paragraph(f"Repository Analysis Report:<br/>{repo_name} ({role.title()} Perspective)", title_style))
        flowables.append(Spacer(1, 24))

        # Table of Contents
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(fontSize=12, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceAfter=6)
        ]
        flowables.append(Paragraph("Table of Contents", heading1))
        flowables.append(Spacer(1, 12))
        flowables.append(toc)
        flowables.append(PageBreak())

        # Key Findings
        if key_findings:
            flowables.append(Paragraph("Key Findings", heading1))
            flowables.append(Spacer(1, 12))
            for point in key_findings:
                bullet = f"• {point.strip()}"
                flowables.append(Paragraph(bullet, normal))
                flowables.append(Spacer(1, 6))
            flowables.append(Spacer(1, 24))

        # Section headers + narrative
        section_headers = ROLE_SECTION_HEADERS.get(role.lower(), ["Narrative"])
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunk_size = max(1, len(paragraphs) // len(section_headers))

        # ✅ Setup afterFlowable hook
        def capture_toc_entry(flowable):
            if isinstance(flowable, Paragraph) and hasattr(flowable, '_header_text'):
                doc.notify('TOCEntry', (flowable._level, flowable._header_text, doc.page))

        doc.afterFlowable = capture_toc_entry

        for i, header in enumerate(section_headers):
            # Add header paragraph
            section_para = Paragraph(header, heading1)
            section_para._header_text = header  # Mark for TOC
            section_para._level = 0
            flowables.append(section_para)
            flowables.append(Spacer(1, 6))

            # Add section content
            for para in paragraphs[i * chunk_size: (i + 1) * chunk_size]:
                cleaned_para = remove_img_tags(para)
                flowables.append(Paragraph(cleaned_para, normal))
                flowables.append(Spacer(1, 12))

        doc.build(flowables, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        return output_path


    def _add_page_number(self, canvas, doc):
        page_num_text = f"Page {doc.page}"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(200 * mm, 15 * mm, page_num_text)


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
