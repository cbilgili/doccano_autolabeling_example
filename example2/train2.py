# coding: utf-8
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, GRU
from keras.layers import Dropout
from keras.layers.convolutional import SeparableConv1D, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.metrics import binary_accuracy
import keras.backend as K
import sentencepiece as spm
import pickle
import numpy as np
import sqlite3
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
import tensorflow as tf
import tensorflow_hub as hub


def build_model(num_class):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(200,)))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_class, activation='sigmoid', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[binary_accuracy])
    return model 


def load_labels(labelfile="/doccano/labels_categorized.json"):
    label2ind = []
    with open(labelfile) as f:
        labels = json.load(f)
        print(labels)
        label2ind = {int(x[0]):int(x[1][0]) for x in labels}
    print(label2ind)
    return label2ind


def prepare(
        labelfile="/doccano/labels_categorized.json",
        dbfile="/doccano/app/db.sqlite3",
        spfile="/doccano/sp_model/jawiki.model"
):
    label2ind = load_labels(labelfile)
    conn = sqlite3.connect(dbfile)
    sp = spm.SentencePieceProcessor()
    sp.Load(spfile)
    return label2ind, conn, sp


def label2vec(label, label2ind):
    out = np.zeros(len(label2ind.keys()))
    out[label2ind[label]] = 1.0
    return out

def calc_vecs(model, X):
    vecs = []
    for xs in X:
        tmp = []
        for x in xs:
            try:
                tmp.append(model.wv[x])
            except:
                continue
        if tmp:
            vecs.append(np.mean(tmp, axis=0))
        else:
            vecs.append(np.zeros(200))
    return vecs

def preprocess(textdata, labeldata, textdata_test, labeldata_test, label2ind):
    import MeCab
    from gensim.models import KeyedVectors
    import numpy as np

    model = KeyedVectors.load("/doccano/w2v.kv", mmap="r")
    tagger = MeCab.Tagger("-Owakati")
    indices = list(textdata.keys())
    indices_test = list(textdata_test.keys())
    X = [textdata[i] for i in indices]
    X_test = [textdata_test[i] for i in indices_test]
    y = [np.sum([label2vec(label_id, label2ind) for label_id in labeldata[i]], axis=0) for i in indices]
    y_test = [np.sum([label2vec(label_id, label2ind) for label_id in labeldata_test[i]], axis=0) for i in indices_test]
    y = np.array(y)
    y_test = np.array(y_test)
    X_fix = [tagger.parse(str(x)).split() for x in X]
    X_fix_test = [tagger.parse(str(x)).split() for x in X_test]
    vecs = calc_vecs(model, X_fix)
    X_fix = np.array(vecs)
    X_test = np.array(calc_vecs(model, X_fix_test))
    X, y = shuffle(X_fix, y) 
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def search(conn, project_ids=[1,3]):
    textdata = {}
    labeldata = defaultdict(list)
    textdata_test = {}
    labeldata_test = defaultdict(list)
    c = conn.cursor()
    for project_id in project_ids:
        c.execute("""
        select server_document.id, text, label_id 
        from server_document 
        inner join server_documentannotation 
        on server_document.id = server_documentannotation.document_id 
        where project_id=?
        """, (project_id,))
        labels = c.fetchall()
        for t in labels:
            if project_id == 1:
                textdata[int(t[0])] = str(t[1])
                labeldata[int(t[0])].append(int(t[2]))
            else:
                textdata_test[int(t[0])] = str(t[1])
                labeldata_test[int(t[0])].append(int(t[2]))
                
    assert(sorted(list(textdata.keys())) == sorted(list(labeldata.keys())))
    return textdata, labeldata, textdata_test, labeldata_test
   
if __name__ == "__main__":
    import pickle
    from sklearn.metrics import classification_report, roc_auc_score
    label2ind, conn, sp = prepare()
    textdata, labeldata, textdata_test, labeldata_test = search(conn)
    train, val, test = preprocess(textdata, labeldata, textdata_test, labeldata_test, label2ind)
    model = build_model(len(label2ind.keys()))
    mcp_save = ModelCheckpoint('/doccano/model2.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(
        *train,
        validation_data=val, epochs=15, batch_size=50, verbose=1, callbacks=[mcp_save])

    preds = model.predict(test[0])

    with open("testdata.pkl", "wb") as f:
        pickle.dump({"data":test, "preds": preds}, f)

    y_trues = []
    y_preds = []
    for ts, ps in zip(test[1], preds):
        for t, p in zip(ts, ps):
            y_trues.append(t > 0.5)
            y_preds.append(p > 0.5)

    result = classification_report(y_trues, y_preds)

    with open("report.txt", "w") as f:
        f.write(result)
        f.write("AUC:"+str(roc_auc_score(y_trues, y_preds)))
            
                
