"""PDF parser implementations."""

import fitz
from llama_index.core.schema import Document

from src.observability import observe
from src.pipeline import Parser as ParserABC


class PyMuPDFParser(ParserABC):
    @observe(as_type="span")
    def parse(self, file_path: str) -> list[Document]:
        docs: list[Document] = []
        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "total_pages": len(pdf),
                        },
                    )
                )
        return docs


__all__ = ["PyMuPDFParser"]
