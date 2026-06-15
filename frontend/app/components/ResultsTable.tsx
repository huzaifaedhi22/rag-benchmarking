const METHOD_ORDER = ["BM25", "Dense", "Hybrid", "ColBERT", "Dense+Reranker", "Dense+ReasoningReranker", "QueryRewriting+Hybrid"];

export type Result = {
  method: string;
  "ndcg@10": number;
  "recall@10": number;
  mrr: number;
  latency_ms: number;
};

type Props = {
  results: Result[];
  phase: "running" | "done";
  chunkCount: number;
  questionCount: number;
  onReset: () => void;
};

export default function ResultsTable({ results, phase, chunkCount, questionCount, onReset }: Props) {
  const currentMethod = METHOD_ORDER[results.length] ?? "";

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <p className="text-xs text-gray-400">
          {chunkCount} chunks · {questionCount > 0 ? `${questionCount} questions` : "generating questions…"}
        </p>
        {phase === "done" && (
          <button onClick={onReset} className="text-xs text-gray-400 hover:text-gray-600 transition-colors cursor-pointer">
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
  );
}
