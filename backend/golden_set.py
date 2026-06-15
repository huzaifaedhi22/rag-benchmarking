import os
import random
from groq import Groq

MIN_CHUNK_CHARS = 300
TARGET_QUESTIONS = 50

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _generate_question(chunk_text: str) -> str | None:
    prompt = (
        "Given the document chunk below, write exactly one question that:\n"
        "- A real user would naturally ask\n"
        "- Can only be answered by reading this specific chunk\n"
        "- Is specific, not generic\n"
        "- Ends with a question mark\n"
        "- Is a single sentence only, no preamble\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        "Question:"
    )

    try:
        response = _get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        question = response.choices[0].message.content.strip()
        return question if question.endswith("?") else None
    except Exception as e:
        print(f"[golden_set] API error: {e}")
        return None


def generate_golden_set(chunks: list[dict], n: int = TARGET_QUESTIONS) -> list[dict]:
    eligible = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]
    sampled = random.sample(eligible, min(n, len(eligible)))

    golden = []
    for i, chunk in enumerate(sampled):
        question = _generate_question(chunk["text"])
        if question:
            golden.append({
                "query_id": f"q_{i}",
                "question": question,
                "relevant_chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
            })

    return golden
