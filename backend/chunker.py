import io
import tiktoken
import pypdf
from dataclasses import dataclass, asdict

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int


def _extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="ignore")


def _tokenize_and_chunk(text: str, source: str, enc: tiktoken.Encoding) -> list[Chunk]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_text = enc.decode(tokens[start:end]).strip()

        if chunk_text:
            slug = source.replace("/", "_").replace(".", "_").replace(" ", "_")
            chunks.append(Chunk(
                chunk_id=f"{slug}_{index}",
                text=chunk_text,
                source=source,
                chunk_index=index,
            ))
            index += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_files(files: list[tuple[str, bytes]]) -> list[dict]:
    enc = tiktoken.get_encoding("cl100k_base")
    all_chunks = []

    for filename, content in files:
        text = _extract_text(filename, content)
        chunks = _tokenize_and_chunk(text, filename, enc)
        all_chunks.extend(chunks)

    return [asdict(c) for c in all_chunks]
