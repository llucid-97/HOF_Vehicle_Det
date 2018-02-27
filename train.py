import time
from loadData import loadAll
from HOG import feature_extract
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pandas
import pickle


def train():
    # load data
    pos_train, neg_train, pos_val, neg_val = loadAll()

    t = time.time()
    hog_pos_train = feature_extract(pos_train)
    hog_neg_train = feature_extract(neg_train)
    t = time.time() - t
    print("Training Feat Extraction time: ", round(t, 2))
    hog_pos_val = feature_extract(pos_val)
    hog_neg_val = feature_extract(neg_val)

    # Concat and generate labels (1 =  Car, 0 =not car)

    X_train = np.vstack((hog_pos_train, hog_neg_train)).astype(np.float64)

    X_train = (X_train - np.mean(X_train)) / np.max(X_train)
    y_train = np.hstack((np.ones(len(hog_pos_train)), np.zeros(len(hog_neg_train))))
    X_train, y_train = shuffle(X_train, y_train)

    X_val = np.vstack((hog_pos_val, hog_neg_val)).astype(np.float64)
    y_val = np.hstack((np.ones(len(hog_pos_val)), np.zeros(len(hog_neg_val))))
    X_val, y_val = shuffle(X_val, y_val)

    # Train SVC
    c = 3.0

    model = LinearSVC(C=1.0, tol=1e-5)
    t = time.time()
    model.fit(X_train, y_train)
    t = time.time() - t

    # predict
    y_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)

    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)

    print("Training took", t, "seconds")
    print("Training accuracy: ", train_acc)
    print("Validation accuracy: ", val_acc)
    return model


def saveModel(model, name="./model/SVM_Classifier.sav"):
    pickle.dump(model, open(name, 'wb'))


def loadModel(name="./model/SVM_Classifier.sav"):
    return pickle.load(open(name, 'rb'))


if __name__ == "__main__":
    model = train()
    saveModel(model)
