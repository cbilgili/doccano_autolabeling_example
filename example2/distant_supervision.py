import sqlite3
import json
from datetime import date
from tqdm import tqdm

def update_db(conn, word, label_text, project_id=1):
    c = conn.cursor()
    #get documents
    sql = "select id from server_document where text like ? and project_id=?"
    c.execute(sql, ('%'+word+'%',project_id))
    indices = c.fetchall()

    #get label_id
    sql ="select id from server_label where text like ? and project_id=?"
    c.execute(sql, ('%'+label_text+'%',project_id))
    label = c.fetchone()

    #insert 'em
    sql = "insert into server_documentannotation(document_id,label_id,prob,user_id,created_at,updated_at,manual) values (?,?,?,?,?,?,?)"
    today = date.today()
    for index in tqdm(indices):
        row = (index[0], label[0], 1.0, 1, today, today, 0)
        try:
            c.execute(sql, row)
        except:
            continue
        conn.commit()

if __name__ == "__main__":
    import sys
    word = sys.argv[1]
    label_text = sys.argv[2]
    conn = sqlite3.connect("/doccano/app/db.sqlite3")
    update_db(conn, word, label_text, project_id=1)
