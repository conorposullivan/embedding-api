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
        return Response('{"response":"Unprocessable Entity"}',
                        status=422, mimetype="application/json")
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
    return {"embeddings": []}


@ app.route("/embedding/similarity", methods=['POST'])
def similarity():
    data = request.get_json()
    return {
        "similarity": 0.28
    }
