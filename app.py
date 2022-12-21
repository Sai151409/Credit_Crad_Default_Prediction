from flask import Flask
import datetime

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} Good Morning to Every one'


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")