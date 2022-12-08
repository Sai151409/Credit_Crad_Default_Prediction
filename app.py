from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    try:
        return 'Good Morning to One and All'
    except Exception as e:
        return e

if __name__ == '__main__' :
    app.run()