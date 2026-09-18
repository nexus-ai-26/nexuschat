"use client";

import { FormEvent, useState } from "react";

interface QuestionFormProps {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

export function QuestionForm({ onSubmit, isLoading }: QuestionFormProps) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed);
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <label htmlFor="question-input" className="font-medium text-gray-700">
        Ask a question about your team&apos;s chat history
      </label>
      <textarea
        id="question-input"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="e.g. What did the team decide about the deployment date?"
        rows={3}
        disabled={isLoading}
        className="w-full resize-none rounded border border-gray-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      />
      <button
        id="ask-button"
        type="submit"
        disabled={isLoading || !value.trim()}
        className="self-end rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? "Searching..." : "Ask"}
      </button>
    </form>
  );
}
