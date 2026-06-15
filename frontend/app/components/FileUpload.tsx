"use client";

import { useRef, useState } from "react";

const SUPPORTED_EXTS = [".pdf", ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".go", ".rs"];

function isSupported(file: File) {
  return SUPPORTED_EXTS.some((ext) => file.name.toLowerCase().endsWith(ext));
}

type Props = {
  onRun: (files: File[]) => void;
};

export default function FileUpload({ onRun }: Props) {
  const [files, setFiles] = useState<File[]>([]);
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

  return (
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
        onClick={() => onRun(files)}
        disabled={!files.length}
        className="mt-5 w-full py-2 bg-gray-900 text-white text-sm font-medium rounded-lg disabled:opacity-25 hover:bg-gray-700 transition-colors cursor-pointer disabled:cursor-default"
      >
        Run benchmark
      </button>
    </div>
  );
}
