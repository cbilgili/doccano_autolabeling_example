# coding: utf-8
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, GRU
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import SeparableConv1D, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import sentencepiece as spm
import pickle
import numpy as np
import sqlite3
import json
from sklearn.model_selection import train_test_split
 
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
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model 

def load_labels(labelfile="/doccano/labels.json"):
    label2ind = []
    with open(labelfile) as f:
        base = json.load(f)["base"]
        for k,v in base.items():
             labels = sorted(list(v.items()), key=lambda x: x[1])
             label2ind += [(int(label[0]),int(i)) for i,label in enumerate(labels)]
        label2ind = dict(label2ind)
    print(label2ind)
    return label2ind

def prepare(
        labelfile="/doccano/labels.json",
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

def preprocess(data, label2ind):
    X = [x[0] for x in data]
    y = [label2vec(x[1], label2ind) for x in data]
    y = np.array(y)
    X_fix = np.array([sp.EncodeAsIds(str(x)) for x in X])
    X_fix = pad_sequences(X_fix, 300) 
    X, y = shuffle(X_fix, y) 
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    return X_train, X_val, y_train, y_val


def search(conn, project_ids=[1]):
    data = []
    c = conn.cursor()
    for project_id in project_ids:
        c.execute("""
        select text, label_id 
        from server_document 
        inner join server_documentannotation 
        on server_document.id = server_documentannotation.document_id 
        where project_id=?
        """, (project_id,))
        data += c.fetchall()
    return data
   
if __name__ == "__main__":
    label2ind, conn, sp = prepare()
    data = search(conn)
    X_train, X_val, y_train, y_val = preprocess(data, label2ind)
    model = build_model(len(label2ind.keys()))
    mcp_save = ModelCheckpoint('/doccano/model.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), epochs=15, batch_size=50, verbose=0, callbacks=[mcp_save])

