interface AnswerBlockProps {
  answer: string | null;
  isLoading: boolean;
}

export function AnswerBlock({ answer, isLoading }: AnswerBlockProps) {
  if (isLoading) {
    return (
      <div
        id="answer-block"
        className="animate-pulse rounded border border-gray-200 bg-gray-100 p-4"
        aria-busy="true"
        aria-label="Loading answer"
      >
        <div className="mb-2 h-4 w-3/4 rounded bg-gray-300" />
        <div className="mb-2 h-4 w-full rounded bg-gray-300" />
        <div className="h-4 w-1/2 rounded bg-gray-300" />
      </div>
    );
  }

  if (!answer) return null;

  return (
    <div
      id="answer-block"
      className="rounded border border-gray-200 bg-white p-4 shadow-sm"
    >
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
        Answer
      </h2>
      <p className="leading-relaxed text-gray-800">{answer}</p>
    </div>
  );
}
