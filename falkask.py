from flask import Flask, jsonify, request

app = Flask(__name__)

# Basic GET route
@app.route('/')
def home():
    return "Hello, Flask!"

# Route with a URL parameter
@app.route('/user/<name>')
def greet(name):
    return f"Hello, {name}!"

# Route that handles JSON data (POST)
@app.route('/api/data', methods=['POST'])
def handle_data():
    data = request.get_json()
    return jsonify({
        "status": "success",
        "received": data
    }), 201

if __name__ == '__main__':
    app.run(debug=True)
