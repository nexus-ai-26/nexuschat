import { Source } from "@/lib/types";
import { CitationCard } from "./CitationCard";

interface CitationListProps {
  sources: Source[];
}

export function CitationList({ sources }: CitationListProps) {
  if (sources.length === 0) {
    return (
      <p id="no-sources" className="text-sm italic text-gray-500">
        No sources cited for this answer.
      </p>
    );
  }

  return (
    <section id="citation-list" aria-label="Sources">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
        Sources ({sources.length})
      </h2>
      <ul className="flex flex-col gap-3">
        {sources.map((source, i) => (
          <li key={source.message_id}>
            <CitationCard source={source} index={i} />
          </li>
        ))}
      </ul>
    </section>
  );
}
