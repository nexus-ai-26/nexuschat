import { ApiResponse } from "@/lib/types";

type Confidence = ApiResponse["confidence"];

const BADGE_STYLES: Record<NonNullable<Confidence>, string> = {
  high: "bg-green-100 text-green-800 border-green-200",
  medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  low: "bg-red-100 text-red-800 border-red-200",
};

interface ConfidenceBadgeProps {
  confidence: Confidence;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  if (!confidence) return null;

  return (
    <span
      id="confidence-badge"
      className={`inline-block rounded border px-2 py-0.5 text-xs font-medium ${BADGE_STYLES[confidence]}`}
    >
      Confidence: {confidence}
    </span>
  );
}
