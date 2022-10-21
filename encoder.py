import tensorflow_hub as hub
from scipy import spatial


class SentenceEncoder():
    def __init__(self, model_path):
        try:
            self.model = hub.load(model_path)
        except:
            raise Exception(
                "Failed to load model. Check the model path is correct")

    def compute_embeddings(self, sentences):
        return self.model(sentences)

    def compute_similarity(self, sentence_one, sentence_two):
        embedding_one = self.compute_embeddings(sentence_one)
        embedding_two = self.compute_embeddings(sentence_two)
        cosine_similarity = 1 - \
            spatial.distance.cosine(embedding_one, embedding_two)
        return cosine_similarity
