import pytest
import requests


def test_get_embeddings_1():
    r = requests.get(
        "http://localhost:5000/embeddings?sentence=the+quick+brown+fox")
    assert r.status_code == 200


def test_get_embedddings_2():
    r = requests.get(
        "http://localhost:5000/embeddings?")
    assert r.status_code == 422


def test_get_embeddings_3():
    r = requests.get(
        "http://localhost:5000/embeddings?sentence=the+quick+brown+fox")
    assert "embedding" in r.json()
