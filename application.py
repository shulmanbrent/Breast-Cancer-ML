import os, csv
import numpy
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
from sklearn import linear_model, tree
from sklearn.externals import joblib
# from pylab import scatter, show, legend, xlabel, ylabel
# from numpy import where
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

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
        clf = joblib.load('log_reg.pkl')
        prediction = prediction_to_diagnosis(clf, inp)


        return render_template("model.html", fields=feature_labels,
                                prediction=prediction,
                                acc=session['model_scores'])
    else:
        features, labels = open_data_file()

        # Build models
        log_reg = LogisticRegression(features, labels)
        dec_tree = DecisionTree(features, labels)
        nn_model = build_nn_model(features, labels)

        # Store models for later evaluation
        joblib.dump(log_reg, 'log_reg.pkl', compress=9)
        joblib.dump(log_reg, 'dec_tree.pkl', compress=9)

        # Dictionary used for displaying accuracy on front end
        model_scores = dict()

        # Get logistic regression accuracy
        log_reg_acc = log_reg.score(features, labels) * 100
        model_scores['Logistic Regression'] = int(log_reg_acc)

        # Get decision tree accuracy
        dec_tree_acc = dec_tree.score(features, labels) * 100
        model_scores['Decision Tree'] = int(dec_tree_acc)

        nn_model_acc = nn_model.evaluate(features, labels, show_accuracy=True, verbose=0)
        print(nn_model_acc)
        model_scores['Neural Network'] = int(nn_model_acc[1] * 100)

        # Save accuracies to session variable
        session['model_scores'] = model_scores
        return render_template("model.html", acc=model_scores,
                                fields=feature_labels)


def prediction_to_diagnosis(classifier, inp):
    if not isinstance(inp, numpy.ndarray):
        inp = numpy.array(inp)
    prediction = classifier.predict(inp)
    if 2 in prediction:
        prediction = 'BENIGN'
    elif 4 in prediction:
        prediction = "MALIGNANT"
    else:
        raise ValueError("Unknown preduction produced")
    return prediction

# def render_graph(features, labels):
#     # BAD! needs to be fixed
#     X = features[:, 0:2]
#     malignant = where(labels == 4)
#     benign = where(labels == 2)
#     scatter(X[malignant, 0], X[malignant, 1], marker='o', c='b')
#     scatter(X[benign, 0], X[benign, 1], marker='x', c='r')
#     xlabel(feature_labels[0])
#     ylabel(feature_labels[1])
#     legend(["Malignant", "Benign"])
#     return show


def open_data_file():
    with open("breastcancer_data.csv", 'r') as data:
        data = csv.reader(data)
        features, labels = convert_to_numpy_array(data)
        return features, labels

def LogisticRegression(features, labels):
    log_reg = linear_model.LogisticRegression()
    log_reg.fit(features, labels)
    return log_reg

def DecisionTree(features, labels):
    dec_tree = tree.DecisionTreeClassifier()
    dec_tree.fit(features, labels)
    return dec_tree

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

def build_nn_model(features, labels):
    model = Sequential()
    model.add(Dense(9, 64, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, 64, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, 1, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(features, labels, nb_epoch=20, batch_size=16)
    return model

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port)
