
import pytest

pytest.importorskip("flask")

from app import app

@pytest.fixture()
def client():
    app.config.update(TESTING=True, SECRET_KEY="test")
    with app.test_client() as client:
        yield client

def payload():
    return {
        "location": "thane",
        "locality_hint": "pokhran road",
        "society": "dosti vihar",
        "property_type": "apartment",
        "bhk": "2",
        "area_sqft": "1200",
        "current_floor": "5",
        "total_floors": "14",
        "bathrooms": "2",
        "balconies": "1",
        "parking": "1",
        "transaction": "resale",
        "furnishing": "semi-furnished",
        "facing": "east",
        "ownership": "freehold",
        "overlooking": "garden/park",
    }

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200

def test_prediction(client):
    r = client.post("/predict", data=payload())
    assert r.status_code == 200
    assert b"Prediction Result" in r.data
