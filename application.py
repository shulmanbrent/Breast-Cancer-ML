import os, csv
import numpy
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
from sklearn import linear_model

app = Flask(__name__, static_url_path='/static')

dat_labels = ["Sample code number",
			  "Clump Thickness",
			  "Uniformity of Cell Size",
			  "Uniformity of Cell Shape",
			  "Marginal Adhesion",
			  "Single Epithelial Cell Size",
			  "Bare Nuclei","Bland Chromatin",
			  "Normal Nucleoli",
			  "Mitoses",
			  "Class"]


@app.route('/')
def main():
    return render_template("index.html")


@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    clf, features, labels = open_data_file()
    acc = clf.score(features, labels)
    return str(acc)


def open_data_file():
    with open("breastcancer_data.csv", 'r') as data:
        data = csv.reader(data)
        features, labels = convert_to_numpy_array(data)
        clf = linear_model.LogisticRegression()
        clf.fit(features, labels)
        return clf, features, labels


def convert_to_numpy_array(data):
    # [1: ] strips off id number of result
    features = list()
    labels = list()
    for row in data:
        if '?' in row:
            for i, elem in enumerate(row):
                if elem == '?':
                    row[i] = '1'
        row = map(int, row)
        features.append(row[1:10])
        labels.append(row[10])
    features = numpy.array(features)
    labels = numpy.array(labels)
    return features, labels


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port)
