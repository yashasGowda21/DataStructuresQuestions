from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Greet(Resource):
    def get(self):
        # Query param: /greet?name=Adarsh
        name = request.args.get('name', 'Guest')
        return {'message': f'Hello, {name}!'}

class Sum(Resource):
    def post(self):
        # JSON body: {"a": 5, "b": 3}
        data = request.get_json(force=True)  # force=True skips content-type check
        a = data.get('a', 0)
        b = data.get('b', 0)
        return {'sum': a + b}

api.add_resource(Greet, '/greet')
api.add_resource(Sum, '/sum')

if __name__ == '__main__':
    app.run(debug=True)
