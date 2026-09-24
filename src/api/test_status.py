from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.status import router


def test_health_does_not_require_db_or_whatsapp():
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
