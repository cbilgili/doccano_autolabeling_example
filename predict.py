# coding: utf-8
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import sentencepiece as spm
import pickle
import numpy as np
import sqlite3
import json
from sklearn.model_selection import train_test_split
 

def load_labels(labelfile="/doccano/labels.json"):
    with open(labelfile) as f:
        v = json.load(f)["ml"]
        labels = sorted(list(v.items()), key=lambda x: x[1])
        ind2label = {int(i):int(label[0]) for i,label in enumerate(labels)}
    return ind2label

def prepare(
        labelfile="/doccano/labels.json",
        dbfile="/doccano/app/db.sqlite3",
        spfile="/doccano/sp_model/jawiki.model",
        modelfile="/doccano/model.h5"
):
    ind2label = load_labels(labelfile)
    conn = sqlite3.connect(dbfile)
    sp = spm.SentencePieceProcessor()
    sp.Load(spfile)
    model = load_model(modelfile)
    return ind2label, conn, sp, model


def preprocess(data):
    X = [str(x[1]) for x in data]
    X_fix = np.array([sp.EncodeAsIds(str(x)) for x in X])
    X_fix = pad_sequences(X_fix, 300) 
    return np.array(X_fix)


def search(conn, project_id=2):
    c = conn.cursor()
    c.execute("""
        select server_document.id, text 
        from server_document
        where project_id=?
    """, (project_id,))
    data = c.fetchall()
    return data


if __name__ == "__main__":
    ind2label, conn, sp, model = prepare()
    data = search(conn)
    X = preprocess(data)
    preds = model.predict_proba(X)
    out = []
    for d, xs in zip(data, preds):
        i = np.argmax(xs)
        out.append((d[0],ind2label[i],float(xs[i])))
    with open("pred.json", "w") as f:
        json.dump(out,f)
