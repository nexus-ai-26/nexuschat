// --- API Contract Types ---
// Schema based on the proposed contract from the technical brief (section 4).
// NOT yet confirmed by Dominique (backend). Treat as a proposal.
// One-liner swap when schema is finalised: update this file and api.ts only.

export type Source = {
  /** WhatsApp message ID -- reserved for future "view source" deep-link */
  message_id: string;
  author: string;
  /** ISO 8601 timestamp */
  timestamp: string;
  excerpt: string;
};

export type ApiResponse = {
  answer: string;
  sources: Source[];
  /** Optional -- useful when semantic search returns a weak match */
  confidence?: "high" | "medium" | "low";
};
