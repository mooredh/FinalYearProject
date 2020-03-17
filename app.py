import flask
from flask import request, jsonify, make_response

app = flask.Flask(__name__)
app.config["DEBUG"] = True

obj = [
    {
        "isPredator": True,
        "accuracy": 0.96
    }
]

@app.route('/api/v1/predators', methods=['POST', 'GET'])
def home():
    return jsonify(obj)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)