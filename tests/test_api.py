import pytest
import requests
import json


@pytest.fixture
def payload_two():
    with open('files/payload_2.json') as json_file:
        payload = json.load(json_file)
    return payload


@pytest.fixture
def payload_three():
    with open('files/payload_3.json') as json_file:
        payload = json.load(json_file)
    return payload


@pytest.fixture
def invalid_payload():
    with open('files/invalid_payload.json') as json_file:
        payload = json.load(json_file)
    return payload


def test_get_embeddings_1():
    r = requests.get(
        "http://localhost:5000/embeddings?sentence=the+quick+brown+fox")
    assert r.status_code == 200


def test_get_embedddings_2():
    r = requests.get(
        "http://localhost:5000/embeddings?")
    assert r.status_code == 400


def test_get_embeddings_3():
    r = requests.get(
        "http://localhost:5000/embeddings?sentence=the+quick+brown+fox")
    assert "embedding" in r.json()


def test_bulk_embeddings_1(payload_two):
    print(payload_two)
    r = requests.post(
        "http://localhost:5000/embeddings/bulk", json=payload_two)
    assert r.status_code == 200


def test_bulk_embeddings_2(payload_two):
    r = requests.post(
        "http://localhost:5000/embeddings/bulk", json=payload_two)
    assert "embeddings" in r.json()


def test_bulk_embeddings_3(payload_two):
    r = requests.post(
        "http://localhost:5000/embeddings/bulk", json=payload_two)
    assert len(r.json()["embeddings"]) == 2


def test_bulk_embeddings_4(invalid_payload):
    print(payload_two)
    r = requests.post(
        "http://localhost:5000/embeddings/bulk", json=invalid_payload)
    assert r.status_code == 400


def test_similarity_1(payload_three):
    r = requests.post(
        "http://localhost:5000/embeddings/similarity", json=payload_three)
    assert r.status_code == 200


def test_similarity_2(payload_three):
    r = requests.post(
        "http://localhost:5000/embeddings/similarity", json=payload_three)
    assert "similarity" in r.json()


def test_similarity_3(payload_three):
    r = requests.post(
        "http://localhost:5000/embeddings/similarity", json=payload_three)
    assert r.json()["similarity"] == 0.28


def test_similarity_4(invalid_payload):
    r = requests.post(
        "http://localhost:5000/embeddings/similarity", json=invalid_payload)
    assert r.status_code == 400
