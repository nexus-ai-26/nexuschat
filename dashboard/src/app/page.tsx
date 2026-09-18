"use client";

import { useState } from "react";
import { QuestionForm } from "@/components/QuestionForm";
import { AnswerBlock } from "@/components/AnswerBlock";
import { CitationList } from "@/components/CitationList";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { queryApi } from "@/lib/api";
import { ApiResponse } from "@/lib/types";

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleQuestion(question: string) {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const data = await queryApi(question);
      setResponse(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unexpected error occurred."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">
          NexusChat Dashboard
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Ask questions about your team&apos;s WhatsApp history. Answers include
          verified sources.
        </p>
      </header>

      <div className="flex flex-col gap-6">
        {/* Block 1 -- Question input */}
        <section id="question-section">
          <QuestionForm onSubmit={handleQuestion} isLoading={isLoading} />
        </section>

        {/* Error state */}
        {error && (
          <p id="error-message" role="alert" className="text-sm text-red-600">
            {error}
          </p>
        )}

        {/* Block 2 -- AI Answer */}
        {(isLoading || response) && (
          <section id="answer-section">
            {response && (
              <div className="mb-2">
                <ConfidenceBadge confidence={response.confidence} />
              </div>
            )}
            <AnswerBlock
              answer={response?.answer ?? null}
              isLoading={isLoading}
            />
          </section>
        )}

        {/* Block 3 -- Citation cards */}
        {response && !isLoading && (
          <section id="sources-section">
            <CitationList sources={response.sources} />
          </section>
        )}
      </div>
    </main>
  );
}
