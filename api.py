from flask import Flask, request, Response
from encoder import SentenceEncoder

# Initialise encoder object
model = SentenceEncoder(
    "models/universal-sentence-encoder_4")

app = Flask(__name__)


@ app.route("/")
def root():
    return {"response": "API is running!"}


@ app.route("/embeddings", methods=["GET"])
def get_embeddings():
    args = request.args
    args = args.to_dict()
    sentence = args.get("sentence", "")
    if not sentence:
        return Response('{"response":"Bad Request"}',
                        status=400, mimetype="application/json")
    try:
        embedding = model.compute_embeddings([sentence])
        embedding_lst = embedding.numpy().tolist()[0]
    except:
        return Response('{"response":"Internal Server Error"}',
                        status=500, mimetype="application/json")
    return {"embedding": embedding_lst}


@ app.route("/embeddings/bulk", methods=['POST'])
def bulk_embeddings():
    data = request.get_json()
    sentences = data.get("sentences", [])
    if not sentences:
        return Response('{"response":"Bad Request"}',
                        status=400, mimetype="application/json")
    try:
        embeddings = model.compute_embeddings(sentences)
        embedding_lst = embeddings.numpy().tolist()
    except:
        return Response('{"response":"Internal Server Error"}',
                        status=500, mimetype="application/json")
    return {"embeddings": embedding_lst}


@ app.route("/embeddings/similarity", methods=['POST'])
def similarity():
    data = request.get_json()
    sentence_1 = data.get("sentence_1", "")
    sentence_2 = data.get("sentence_2", "")
    if not sentence_1 or not sentence_2:
        return Response('{"response":"Bad Request"}',
                        status=400, mimetype="application/json")
    try:
        similarity = model.compute_similarity([sentence_1], [sentence_2])
    except:
        return Response('{"response":"Internal Server Error"}',
                        status=500, mimetype="application/json")
    return {
        "similarity": similarity
    }
