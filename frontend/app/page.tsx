"use client";

import { useState } from "react";
import FileUpload from "./components/FileUpload";
import ResultsTable, { type Result } from "./components/ResultsTable";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_SECRET = process.env.NEXT_PUBLIC_API_SECRET;

type Phase = "idle" | "uploading" | "running" | "done" | "error";

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [results, setResults] = useState<Result[]>([]);
  const [chunkCount, setChunkCount] = useState(0);
  const [questionCount, setQuestionCount] = useState(0);
  const [error, setError] = useState("");

  async function run(files: File[]) {
    setPhase("uploading");
    setResults([]);
    setError("");

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));

      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: form,
        headers: API_SECRET ? { "x-api-secret": API_SECRET } : {},
      });
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
    setResults([]);
    setError("");
    setChunkCount(0);
    setQuestionCount(0);
  }

  return (
    <main className="min-h-screen bg-white">
      <div className="max-w-2xl mx-auto px-6 py-20">

        <div className="mb-14">
          <h1 className="text-xl font-semibold tracking-tight text-gray-900 mb-1">RAG Benchmark</h1>
          <p className="text-sm text-gray-400">Upload your documents. We test 7 retrieval methods and tell you which one wins on your data.</p>
        </div>

        {(phase === "idle" || phase === "error") && (
          <>
            <FileUpload onRun={run} />
            {error && <p className="mt-3 text-xs text-red-400">{error}</p>}
          </>
        )}

        {phase === "uploading" && (
          <p className="text-sm text-gray-400">Uploading and chunking documents…</p>
        )}

        {(phase === "running" || phase === "done") && (
          <ResultsTable
            results={results}
            phase={phase}
            chunkCount={chunkCount}
            questionCount={questionCount}
            onReset={reset}
          />
        )}

      </div>
    </main>
  );
}
