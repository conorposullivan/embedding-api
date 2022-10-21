import pytest
from encoder import SentenceEncoder


@pytest.fixture
def encoder():
    return SentenceEncoder("models/universal-sentence-encoder_4")


def test_sentence_encoder():
    with pytest.raises(Exception):
        SentenceEncoder("")


def test_calc_cosine_similarity_1(encoder):
    lst_one = [1.0, 0.0, 1.0]
    lst_two = [1.0, 0.0, 1.0]
    assert encoder.calc_cosine_similarity(lst_one, lst_two) == 1.0


def test_calc_cosine_similarity_2(encoder):
    lst_one = [1.0, 0.0, 1.0]
    lst_two = [0.0, 1.0, 0.0]
    assert encoder.calc_cosine_similarity(lst_one, lst_two) == 0.0
