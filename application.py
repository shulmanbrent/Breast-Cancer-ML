import os, csv
import numpy
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
from sklearn import linear_model
from sklearn.externals import joblib
from pylab import scatter, show, legend, xlabel, ylabel
from numpy import where

app = Flask(__name__, static_url_path='/static')

app.secret_key = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'

dat_labels = ["Sample code number",
			  "Clump Thickness",
			  "Uniformity of Cell Size",
			  "Uniformity of Cell Shape",
			  "Marginal Adhesion",
			  "Single Epithelial Cell Size",
			  "Bare Nuclei",
              "Bland Chromatin",
			  "Normal Nucleoli",
			  "Mitoses",
			  "Class"]

feature_labels = ["Clump Thickness",
              "Uniformity of Cell Size",
              "Uniformity of Cell Shape",
              "Marginal Adhesion",
              "Single Epithelial Cell Size",
              "Bare Nuclei",
              "Bland Chromatin",
              "Normal Nucleoli",
              "Mitoses"]

@app.route('/')
def main():
    return render_template("index.html")


@app.route("/predictor", methods=["GET", "POST"])
def predictor():

    if request.method == "POST":
        inp = [0 for i in range(9)]
        # Sorting data
        for label, value in request.form.items():
            i = feature_labels.index(label)
            inp[i] = int(value)
        inp = numpy.array(inp)
        clf = joblib.load('my_model.pkl')
        prediction = clf.predict(inp)
        # graph = render_graph(session)
        return render_template("model.html", fields=feature_labels,
                                prediction=str(prediction),
                                acc=session['acc'])
    else:
        clf, features, labels = open_data_file()
        # Save to session variable
        
        joblib.dump(clf, 'my_model.pkl', compress=9)
        acc = clf.score(features, labels)
        acc *= 100
        session['acc'] = int(acc)
        return render_template("model.html", acc=int(acc),
                                fields=feature_labels)


def render_graph(session):
    # BAD! needs to be fixed
    _, features, labels = open_data_file()
    X = features[:, 0:2]
    malignant = where(labels == 4)
    benign = where(labels == 2)
    scatter(X[malignant, 0], X[malignant, 1], marker='o', c='b')
    scatter(X[benign, 0], X[benign, 1], marker='x', c='r')
    xlabel(feature_labels[0])
    ylabel(feature_labels[1])
    legend(["Malignant", "Benign"])
    return show


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
