"""
SEC Filings Ingestion & Table-Aware Markdown Parser.
Extracts SEC 10-K/10-Q filings, isolates financial tables (Item 1A Risk Factors, Item 7 MD&A),
and structures them into HTML/Markdown chunks to avoid cutting financial tables in half.
"""

from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any

class SECTableAwareParser:
    @staticmethod
    def html_table_to_markdown(table_soup: BeautifulSoup) -> str:
        """Converts an HTML table element into a formatted Markdown string."""
        rows = table_soup.find_all('tr')
        if not rows:
            return ""

        markdown_table = []
        for i, row in enumerate(rows):
            cols = row.find_all(['td', 'th'])
            row_cells = [re.sub(r'\s+', ' ', col.get_text()).strip() for col in cols]
            if not any(row_cells):
                continue
            
            markdown_table.append("| " + " | ".join(row_cells) + " |")
            if i == 0:
                # Add header separator line
                markdown_table.append("| " + " | ".join(["---"] * len(row_cells)) + " |")

        return "\n".join(markdown_table)

    @classmethod
    def parse_sec_html(cls, html_content: str, source_doc: str = "SEC_10K.html") -> List[Dict[str, Any]]:
        """
        Parses raw SEC HTML content into text and table chunks with structural metadata.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        chunks = []

        # Find all section headers or tables
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            md_table = cls.html_table_to_markdown(table)
            if len(md_table.strip()) > 20: # Filter out tiny formatting tables
                chunks.append({
                    "chunk_type": "table",
                    "content": md_table,
                    "metadata": {
                        "source": source_doc,
                        "table_id": i + 1,
                        "is_financial_table": True
                    }
                })

        # Process paragraph text
        paragraphs = soup.find_all(['p', 'div'])
        buffer = []
        buffer_len = 0

        for p in paragraphs:
            text = re.sub(r'\s+', ' ', p.get_text()).strip()
            if len(text) < 30 or p.find('table'):
                continue
                
            buffer.append(text)
            buffer_len += len(text)

            # Combine paragraphs into ~1000 character chunks with overlap
            if buffer_len >= 800:
                combined_text = "\n\n".join(buffer)
                chunks.append({
                    "chunk_type": "text",
                    "content": combined_text,
                    "metadata": {
                        "source": source_doc,
                        "is_financial_table": False
                    }
                })
                buffer = buffer[-1:] # Keep last paragraph for context overlap
                buffer_len = len(buffer[0]) if buffer else 0

        if buffer:
            chunks.append({
                "chunk_type": "text",
                "content": "\n\n".join(buffer),
                "metadata": {
                    "source": source_doc,
                    "is_financial_table": False
                }
            })

        return chunks
