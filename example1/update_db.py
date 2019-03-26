import sqlite3
import json
from datetime import date

def run():
    labdata, conn, preds = prepare()
    label_ids = get_labels(labdata)
    delete_doclab(conn, label_ids)
    update_db(conn, preds)

def prepare(
        dbfile="/doccano/app/db.sqlite3",
        labelfile="/doccano/labels.json",
        predfile="/doccano/pred.json"
):
    with open(labelfile) as f:
        labdata = json.load(f)

    conn = sqlite3.connect(dbfile)

    with open(predfile) as f:
        preds = json.load(f)
        
    return labdata, conn, preds


def get_labels(labdata):
    return list(labdata["ml"].keys())


def delete_doclab(conn, label_ids):
    c = conn.cursor()
    sql = "delete from server_documentannotation where label_id=?"
    for label_id in label_ids:
        c.execute(sql, (label_id,))
        conn.commit()
    return True

def update_db(conn, preds):
    c = conn.cursor()
    sql = "insert into server_documentannotation (document_id,label_id,prob,user_id,created_at,updated_at,manual) values (?,?,?,?,?,?,?)"
    for pred in preds:
        pred = tuple(pred + [1,date.today(),date.today(),0])
        c.execute(sql, pred)
        conn.commit()
    return True


if __name__ == "__main__":
    run()
