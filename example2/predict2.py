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
from gensim.models import KeyedVectors
import MeCab

def load_labels(labelfile="/doccano/labelids.json"):
    with open(labelfile) as f:
        ind2label = dict(enumerate([0] + [x[0] for x in json.load(f)]))
    return ind2label

def prepare(
        labelfile="/doccano/labelids.json",
        dbfile="/doccano/app/db.sqlite3",
        w2vfile="/doccano/w2v.kv",
        modelfile="/doccano/model2.h5"
):
    ind2label = load_labels(labelfile)
    conn = sqlite3.connect(dbfile)
    w2v = KeyedVectors.load(w2vfile, mmap="r") 
    model = load_model(modelfile)
    return ind2label, conn, w2v, model


def preprocess(data, w2v):
    tagger = MeCab.Tagger("-Owakati")
    X = [tagger.parse(str(x[1])).split() for x in data]
    vecs = []
    for xs in X:
        tmp = []
        for x in xs:
            try:
                tmp.append(w2v.wv[x])
            except:
                continue
        if tmp:
            vecs.append(np.mean(tmp, axis=0))
        else:
            vecs.append(np.zeros(200))
    X_fix = np.array(vecs)
    return X_fix


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
    threshold = 0.95
    ind2label, conn, w2v, model = prepare()
    data = search(conn)
    X = preprocess(data, w2v)
    preds = model.predict(X)
    out = []
    for d, xs in zip(data, preds):
        for i,x in enumerate(xs[1:]):
            if x < threshold:
                continue
            out.append((d[0],ind2label[i+1],float(x)))
    with open("pred.json", "w") as f:
        json.dump(out,f)
