import os
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
from sklearn import linear_model

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def main():
    return "Hello World"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
