import sqlite3
import json
from datetime import date

def run():
    conn, preds = prepare()
    delete_doclab(conn)
    update_db(conn, preds)

def prepare(
        dbfile="/doccano/app/db.sqlite3",
        predfile="/doccano/pred.json"
):

    conn = sqlite3.connect(dbfile)
    with open(predfile) as f:
        preds = json.load(f)
    return conn, preds

def delete_doclab(conn, ml_project=2):
    c = conn.cursor()
    sql = "select id from server_document where project_id=?"
    c.execute(sql, (ml_project,))
    for x in c.fetchall():
        sql = "delete from server_documentannotation where document_id=?"
        c.execute(sql, (x[0],))
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
