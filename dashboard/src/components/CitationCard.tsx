import { Source } from "@/lib/types";

interface CitationCardProps {
  source: Source;
  index: number;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function CitationCard({ source, index }: CitationCardProps) {
  return (
    <article
      id={`citation-card-${index}`}
      className="rounded border border-gray-200 bg-white p-4 shadow-sm"
    >
      <div className="mb-2 flex items-center justify-between text-xs text-gray-500">
        <span className="font-semibold text-gray-700">{source.author}</span>
        <time dateTime={source.timestamp}>
          {formatTimestamp(source.timestamp)}
        </time>
      </div>
      <blockquote className="border-l-2 border-blue-400 pl-3 text-sm italic text-gray-600">
        {source.excerpt}
      </blockquote>
      {/* message_id preserved for future "view source" deep-link */}
      <span className="sr-only">Message ID: {source.message_id}</span>
    </article>
  );
}
