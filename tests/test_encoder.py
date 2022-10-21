import pytest
from opentable.encoder import SentenceEncoder


def test_sentence_encoder():
    with pytest.raises(Exception):
        SentenceEncoder("")


def test_compute_similarity():
    assert 0 == 0.28
