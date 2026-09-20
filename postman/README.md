# Test NexusChat with Postman

Use the Postman desktop application for this local backend. If using Postman in a browser, select its Desktop Agent so requests can reach localhost.

1. Import `NexusChat.postman_collection.json`.
2. Open the imported **NexusChat — Local backend (AI disabled)** collection.
3. Keep the collection variable `base_url` set to `http://localhost:8000`.
4. Open each request and click **Send**, then inspect **Test Results**. You can also run all four requests with the collection runner.

No authorization or request body is needed. This collection targets NexusChat's FastAPI backend, not GoWA on port 3003.

| Request | Expected result |
| --- | --- |
| `GET /readiness` | HTTP 200, `status: ok` |
| `GET /status` | HTTP 200; overall, database, and WhatsApp statuses are healthy |
| `POST /load_new_kbtopics` | HTTP 503 explaining that AI features are disabled |
| `POST /summarize_and_send_to_groups` | HTTP 503 explaining that AI features are disabled |

The two POST tests require `AI_ENABLED=false` in the running backend. HTTP 503 is intentional for these tests. Once AI is enabled, those endpoints perform real ingestion or summary delivery, so stop using this AI-disabled collection unchanged.

These checks verify HTTP availability, service connectivity, and disabled AI behavior. They do not prove that real incoming WhatsApp messages are saved or that Gemini generates answers. To test incoming messages separately, send a normal text from another WhatsApp account to a test group containing the connected account, then check the `message` table. AI-disabled mode stores messages but does not generate AI replies.

If a request cannot connect, confirm the backend is running and that you are using port 8000. A 503 from `/status` is a failed health check; its response identifies the failing service. A 404 usually means the URL or port is incorrect. API documentation is at `http://localhost:8000/docs`.
