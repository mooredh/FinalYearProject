import flask
from flask import request, jsonify, make_response, render_template
import json
from werkzeug.exceptions import HTTPException

app = flask.Flask(__name__)
app.config["DEBUG"] = True

obj = [
    {
        "isPredator": True,
        "accuracy": 0.96
    }
]

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/api/v1/detect', methods=['POST'])
def conversations():
    return jsonify(obj)

@app.errorhandler(HTTPException)
def handle_exception(e):
    response = e.get_response()
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)