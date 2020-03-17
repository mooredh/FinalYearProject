import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

books = [
    {
        "name": "Moore",
        "age": "21"
    }
]

@app.route('/', methods=['GET'])
def home():
    return jsonify(books)

app.run()