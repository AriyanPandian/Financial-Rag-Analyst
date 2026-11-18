import unittest

from app.documents import TextBlock, chunk_blocks, parse_document
from app.errors import AppError


class DocumentTests(unittest.TestCase):
    def test_pdf_chunks_keep_page_citations(self):
        import pymupdf

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Revenue increased to $42 million.")
        content = pdf.tobytes()
        pdf.close()

        chunks = chunk_blocks(parse_document("annual-report.pdf", content))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page, 1)
        self.assertIn("$42 million", chunks[0].text)

    def test_html_tables_and_section_headings_are_preserved(self):
        html = b"""
        <html><body><h2>Item 7. Results</h2><p>Revenue was reported.</p>
        <table><tr><th>Year</th><th>Revenue</th></tr>
        <tr><td>2025</td><td>$42 million</td></tr></table></body></html>
        """

        chunks = chunk_blocks(parse_document("filing.html", html))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Item 7. Results")
        self.assertIn("Year | Revenue", chunks[0].text)
        self.assertIn("2025 | $42 million", chunks[0].text)

    def test_chunks_overlap_without_losing_source_metadata(self):
        text = " ".join(f"token{index}" for index in range(120))

        chunks = chunk_blocks([TextBlock(text, page=3)], chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page == 3 for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 120 for chunk in chunks))
        self.assertTrue(
            any(
                set(chunks[index].text.split())
                & set(chunks[index + 1].text.split())
                for index in range(len(chunks) - 1)
            )
        )

    def test_rejects_unsupported_document_type(self):
        with self.assertRaises(AppError):
            parse_document("filing.xlsx", b"")


if __name__ == "__main__":
    unittest.main()
