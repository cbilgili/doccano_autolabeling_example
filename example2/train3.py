# coding: utf-8
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
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
from tqdm import tqdm

def build_model():
    model = Sequential()
    model.add(Dense(400, activation='relu', input_shape=(400,)))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    return model 

def load_labels(labelfile="/doccano/labels_categorized.json"):
    label2ind = []
    with open(labelfile) as f:
        labels = json.load(f)
        print(labels)
        label2ind = {int(x[0]):int(x[1][0]) for x in labels}
        ind2name = {int(x[1][0]):str(x[1][1]) for x in labels}
    print(label2ind)
    return label2ind, ind2name


def prepare(
        labelfile="/doccano/labels_categorized.json",
        dbfile="/doccano/app/db.sqlite3",
        spfile="/doccano/sp_model/jawiki.model"
):
    label2ind, ind2name = load_labels(labelfile)
    conn = sqlite3.connect(dbfile)
    sp = spm.SentencePieceProcessor()
    sp.Load(spfile)
    return label2ind, ind2name, conn, sp


def label2vec(label, label2ind):
    out = np.zeros(len(label2ind.keys()))
    out[label2ind[label]] = 1
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

def preprocess(textdata, labeldata, textdata_test, labeldata_test, label2ind, tagger, model):
    indices = list(textdata.keys())
    indices_test = list(textdata_test.keys())
    X = [textdata[i] for i in indices]
    y = [np.sum([label2vec(label_id, label2ind) for label_id in labeldata[i]], axis=0) for i in indices]
    y = np.array(y)
    X_fix = [tagger.parse(str(x)).split() for x in X]
    vecs = calc_vecs(model, X_fix)
    X_fix = np.array(vecs)
    X, y = shuffle(X_fix, y) 
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def data_fix(X, y, ind2name, w2v, tagger):
    out_X, out_y = [], []
    embs = {}
    for a,bs in tqdm(zip(X,y)):
        targets = np.where(bs == 1)[0].tolist()
        for target in targets:
            if target not in embs:
                name = ind2name[target]
                tmp = np.mean([w2v.wv[x] for x in tagger.parse(name).split()], axis=0)
                embs[target] = tmp
            else:
                tmp = embs[target]
            for i,b in enumerate(bs[1:]):
                out_X.append(np.hstack((a,tmp)))
                if i+1 == target:
                    out_y.append(True)
                else:
                    out_y.append(False)
    return out_X, out_y

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


def generate(X,y, size=50):
    while True:
        X_tmp, y_tmp = shuffle(X,y)
        yield np.array(X_tmp[:size]), np.array(y_tmp[:size])


if __name__ == "__main__":
    import pickle
    from sklearn.metrics import classification_report, roc_auc_score
    import MeCab
    from gensim.models import KeyedVectors
    import numpy as np

    w2v = KeyedVectors.load("/doccano/w2v.kv", mmap="r")
    tagger = MeCab.Tagger("-Owakati")

    label2ind, ind2name, conn, sp = prepare()
    textdata, labeldata, textdata_test, labeldata_test = search(conn)
    train, val, test = preprocess(textdata, labeldata, textdata_test, labeldata_test, label2ind, tagger, w2v)
    train = data_fix(*train, ind2name, w2v, tagger)
    val = data_fix(*val, ind2name, w2v, tagger)
    test = data_fix(*test, ind2name, w2v, tagger)
    model = build_model()
    mcp_save = ModelCheckpoint('/doccano/model3.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit_generator(
        generate(*train),
        validation_data=(np.array(val[0][:3000]), np.array(val[1][:3000])), steps_per_epoch=1000, epochs=15, verbose=1, callbacks=[mcp_save])

    preds = model.predict_classes(np.array(test[0]))
    result = classification_report(np.array(test[1]), preds)

    with open("report3.txt", "w") as f:
        f.write(result)
        f.write("AUC:"+str(roc_auc_score(np.array(test[1]), preds)))
            
                
