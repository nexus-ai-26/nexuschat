import { ApiResponse } from "./types";

// --- Mock toggle ---
// Set USE_MOCK=false and set NEXT_PUBLIC_API_URL in .env.local to use the real
// backend endpoint. The switch is a single boolean -- no other code changes needed.
//
// NOTE: the API schema (/ask endpoint) is a proposal from the brief (section 4).
//       Confirm the contract with Dominique before flipping this to false.
const USE_MOCK = true;

// --- Fixture data ---
// Representative mock covering: multi-source answer + confidence field.
const MOCK_RESPONSE: ApiResponse = {
  answer:
    "The team decided to deploy on September 21 using docker-compose.prod.yml on Railway.",
  sources: [
    {
      message_id: "wamid.mock-001",
      author: "Dominique",
      timestamp: "2026-09-17T14:32:00Z",
      excerpt: "Deploy to Railway/Render using docker-compose.prod.yml.",
    },
    {
      message_id: "wamid.mock-002",
      author: "Gediyon",
      timestamp: "2026-09-17T15:10:00Z",
      excerpt: "Let us target September 21 for the live deployment deadline.",
    },
  ],
  confidence: "high",
};

// --- API client ---

/**
 * Sends a question to the NexusChat API and returns the AI answer with sources.
 *
 * Mock mode  (USE_MOCK=true):  returns MOCK_RESPONSE after 800ms simulated delay.
 * Real mode  (USE_MOCK=false): POSTs to NEXT_PUBLIC_API_URL/ask.
 */
export async function queryApi(question: string): Promise<ApiResponse> {
  if (USE_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 800));
    return MOCK_RESPONSE;
  }

  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (!apiUrl) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not defined. Add it to .env.local and restart the dev server."
    );
  }

  const res = await fetch(`${apiUrl}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }

  return res.json() as Promise<ApiResponse>;
}
