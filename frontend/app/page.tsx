"use client";

import { useRef, useState } from "react";

const SUPPORTED_EXTS = [".pdf", ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".go", ".rs"];
const METHOD_ORDER = ["BM25", "Dense", "Hybrid", "ColBERT", "Dense+Reranker", "Dense+ReasoningReranker", "QueryRewriting+Hybrid"];
const API = "http://localhost:8000";

type Result = {
  method: string;
  "ndcg@10": number;
  "recall@10": number;
  mrr: number;
  latency_ms: number;
};

type Phase = "idle" | "uploading" | "running" | "done" | "error";

function isSupported(file: File) {
  return SUPPORTED_EXTS.some((ext) => file.name.toLowerCase().endsWith(ext));
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [chunkCount, setChunkCount] = useState(0);
  const [questionCount, setQuestionCount] = useState(0);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function pickFiles(fileList: FileList) {
    const valid = Array.from(fileList).filter(isSupported);
    const invalid = Array.from(fileList).length - valid.length;
    if (invalid > 0) setError(`${invalid} unsupported file${invalid > 1 ? "s" : ""} ignored. Accepted: ${SUPPORTED_EXTS.join(", ")}`);
    else setError("");
    if (valid.length) setFiles(valid);
  }

  async function run() {
    if (!files.length) return;
    setPhase("uploading");
    setResults([]);
    setError("");

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));

      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Upload failed");
      const { job_id, chunk_count } = await res.json();
      setChunkCount(chunk_count);
      setPhase("running");

      const source = new EventSource(`${API}/benchmark/${job_id}`);

      source.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === "start") {
          setQuestionCount(data.question_count);
        } else if (data.type === "result") {
          const { type: _, ...result } = data;
          setResults((prev) => [...prev, result as Result]);
        } else if (data.type === "done") {
          setResults(data.results);
          setPhase("done");
          source.close();
        } else if (data.type === "error") {
          setError(data.message);
          setPhase("error");
          source.close();
        }
      };

      source.onerror = () => {
        setError("Connection to server lost.");
        setPhase("error");
        source.close();
      };
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setPhase("error");
    }
  }

  function reset() {
    setPhase("idle");
    setFiles([]);
    setResults([]);
    setError("");
    setChunkCount(0);
    setQuestionCount(0);
  }

  const currentMethod = METHOD_ORDER[results.length] ?? "";

  return (
    <main className="min-h-screen bg-white">
      <div className="max-w-2xl mx-auto px-6 py-20">

        <div className="mb-14">
          <h1 className="text-xl font-semibold tracking-tight text-gray-900 mb-1">RAG Benchmark</h1>
          <p className="text-sm text-gray-400">Upload your documents. We test 7 retrieval methods and tell you which one wins on your data.</p>
        </div>

        {(phase === "idle" || phase === "error") && (
          <div>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); pickFiles(e.dataTransfer.files); }}
              onClick={() => inputRef.current?.click()}
              className={`border border-dashed rounded-lg px-8 py-14 text-center cursor-pointer transition-colors ${
                dragging ? "border-gray-400 bg-gray-50" : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                multiple
                accept={SUPPORTED_EXTS.join(",")}
                onChange={(e) => e.target.files && pickFiles(e.target.files)}
                className="hidden"
              />
              <p className="text-sm font-medium text-gray-700 mb-1">
                {files.length ? `${files.length} file${files.length > 1 ? "s" : ""} selected` : "Drop files here or click to browse"}
              </p>
              <p className="text-xs text-gray-400">{SUPPORTED_EXTS.join("  ")}</p>
            </div>

            {files.length > 0 && (
              <div className="mt-3 space-y-1">
                {files.map((f) => (
                  <div key={f.name} className="flex justify-between text-xs text-gray-400 px-1">
                    <span className="truncate mr-4">{f.name}</span>
                    <span className="shrink-0">{(f.size / 1024).toFixed(0)} KB</span>
                  </div>
                ))}
              </div>
            )}

            {error && <p className="mt-3 text-xs text-red-400">{error}</p>}

            <button
              onClick={run}
              disabled={!files.length}
              className="mt-5 w-full py-2 bg-gray-900 text-white text-sm font-medium rounded-lg disabled:opacity-25 hover:bg-gray-700 transition-colors cursor-pointer disabled:cursor-default"
            >
              Run benchmark
            </button>
          </div>
        )}

        {phase === "uploading" && (
          <p className="text-sm text-gray-400">Uploading and chunking documents…</p>
        )}

        {(phase === "running" || phase === "done") && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <p className="text-xs text-gray-400">
                {chunkCount} chunks · {questionCount > 0 ? `${questionCount} questions` : "generating questions…"}
              </p>
              {phase === "done" && (
                <button onClick={reset} className="text-xs text-gray-400 hover:text-gray-600 transition-colors cursor-pointer">
                  Run again
                </button>
              )}
            </div>

            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100">
                  <th className="text-left pb-2 text-xs font-medium text-gray-400">Method</th>
                  <th className="text-right pb-2 text-xs font-medium text-gray-400">nDCG@10</th>
                  <th className="text-right pb-2 text-xs font-medium text-gray-400">Recall@10</th>
                  <th className="text-right pb-2 text-xs font-medium text-gray-400">MRR</th>
                  <th className="text-right pb-2 text-xs font-medium text-gray-400">Latency</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={r.method} className="border-b border-gray-50">
                    <td className="py-3 pr-6 text-gray-900">
                      <span className="font-medium">{r.method}</span>
                      {phase === "done" && i === 0 && (
                        <span className="ml-2 text-[10px] font-medium bg-gray-900 text-white px-1.5 py-0.5 rounded">best</span>
                      )}
                    </td>
                    <td className="py-3 pr-6 text-right tabular-nums text-gray-700">{r["ndcg@10"].toFixed(4)}</td>
                    <td className="py-3 pr-6 text-right tabular-nums text-gray-700">{r["recall@10"].toFixed(4)}</td>
                    <td className="py-3 pr-6 text-right tabular-nums text-gray-700">{r.mrr.toFixed(4)}</td>
                    <td className="py-3 text-right tabular-nums text-gray-400">{r.latency_ms}ms</td>
                  </tr>
                ))}
                {phase === "running" && (
                  <tr>
                    <td className="py-3 text-xs text-gray-400" colSpan={5}>
                      <span className="inline-flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-gray-300 animate-pulse" />
                        {currentMethod ? `Running ${currentMethod}…` : "Finishing up…"}
                      </span>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </main>
  );
}
