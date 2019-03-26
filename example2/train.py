# coding: utf-8
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, GRU
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import SeparableConv1D, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import sentencepiece as spm
import pickle
import numpy as np
import sqlite3
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict


def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)


def build_model(num_class, max_features=15000, dim=200, max_len=300, dropout_rate=0.2, gru_size=100):
    model = Sequential()
    model.add(Embedding(max_features+1, dim, input_length=max_len))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(SeparableConv1D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(SeparableConv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(gru_size))
    model.add(Dense(num_class, activation='sigmoid', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[total_acc, binary_acc])
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


def preprocess(textdata, labeldata, label2ind):
    indices = list(textdata.keys())
    X = [textdata[i] for i in indices]
    y = [np.sum([label2vec(label_id, label2ind) for label_id in labeldata[i]], axis=0) for i in indices]
    print(y[0])
    y = np.array(y)
    X_fix = np.array([sp.EncodeAsIds(str(x)) for x in X])
    X_fix = pad_sequences(X_fix, 300) 
    X, y = shuffle(X_fix, y) 
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def search(conn, project_ids=[1,3]):
    textdata = {}
    labeldata = defaultdict(list)
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
            textdata[int(t[0])] = str(t[1])
            labeldata[int(t[0])].append(int(t[2]))
    assert(sorted(list(textdata.keys())) == sorted(list(labeldata.keys())))
    return textdata, labeldata
   
if __name__ == "__main__":
    import pickle
    label2ind, conn, sp = prepare()
    textdata, labeldata = search(conn)
    train, val, test = preprocess(textdata, labeldata, label2ind)
    model = build_model(len(label2ind.keys()))
    mcp_save = ModelCheckpoint('/doccano/model.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(
        *train,
        validation_data=val, epochs=15, batch_size=50, verbose=1, callbacks=[mcp_save])

    preds = model.predict(test[0])
    accs = (float(total_acc(test[1], preds)), float(binary_acc(test[1], preds)))
    
    with open("acc.json", "w") as f:
        json.dump(accs, f)

    with open("testdata.pkl", "wb") as f:
        pickle.dump({"data":test, "preds": preds})
        
    
