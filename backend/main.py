import json
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .chunker import chunk_files
from .golden_set import generate_golden_set
from .eval.harness import stream_benchmark

app = FastAPI(title="RAG Benchmark")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, list[dict]] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    file_data = [(f.filename or "file", await f.read()) for f in files]
    chunks = chunk_files(file_data)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from uploaded files")

    job_id = uuid.uuid4().hex
    _sessions[job_id] = chunks
    return {"job_id": job_id, "chunk_count": len(chunks)}


@app.get("/benchmark/{job_id}")
async def benchmark(job_id: str):
    chunks = _sessions.get(job_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        golden = generate_golden_set(chunks)
        if not golden:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Golden set generation failed'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'start', 'chunk_count': len(chunks), 'question_count': len(golden)})}\n\n"

        all_results = []
        for result in stream_benchmark(chunks, golden):
            all_results.append(result)
            yield f"data: {json.dumps({'type': 'result', **result})}\n\n"

        sorted_results = sorted(all_results, key=lambda x: x["ndcg@10"], reverse=True)
        yield f"data: {json.dumps({'type': 'done', 'results': sorted_results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
