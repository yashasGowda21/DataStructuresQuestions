from flask import Flask, request, jsonify

app = Flask(__name__)
data = {}

@app.route('/items', methods=['POST'])
def create_item():
    item = request.json
    item_id = str(len(data) + 1)
    data[item_id] = item
    return jsonify({"id": item_id, "item": item}), 201

@app.route('/items/<item_id>', methods=['GET'])
def get_item(item_id):
    item = data.get(item_id)
    if item:
        return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

@app.route('/items/<item_id>', methods=['PUT'])
def update_item(item_id):
    if item_id not in data:
        return jsonify({"error": "Item not found"}), 404
    data[item_id] = request.json
    return jsonify({"id": item_id, "item": data[item_id]})

@app.route('/items/<item_id>', methods=['DELETE'])
def delete_item(item_id):
    if item_id in data:
        del data[item_id]
        return jsonify({"message": "Deleted"})
    return jsonify({"error": "Item not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
