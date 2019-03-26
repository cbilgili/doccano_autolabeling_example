import json
import sqlite3
from datetime import date


def connect(dbpath="/doccano/app/db.sqlite3"):
    return sqlite3.connect(dbpath)


def load_labels(labelfile="/doccano/labels_categorized.json"):
    with open(labelfile) as f:
        labels = {int(x[1][0]):x[1][1] for x in json.load(f)}
    labels = sorted([x for x in labels.items()], key=lambda x: x[0])
    return labels


def update_labels(conn, labels, ml_project=2):
    c = conn.cursor()
    labels = [x[1] for x in labels]
    c.execute("delete from server_label where project_id=?", (ml_project,))
    conn.commit()
    today = date.today()
    for label in labels:
        row = (label, "#209cee","#ffffff",ml_project,today,today)
        c.execute("""
        insert into server_label(text,background_color,text_color,project_id,created_at,updated_at)
        values (?,?,?,?,?,?)
        """, row)
        conn.commit()
    c.execute("select id, text from server_label where project_id=?", (ml_project,))
    return sorted(c.fetchall(), key=lambda x: x[0])

if __name__ == "__main__":
    conn = connect()
    labels = load_labels()
    print(labels)
    labels2 = update_labels(conn, labels)
    print(labels2)
    with open("labelids.json", "w") as f:
        json.dump(labels2, f)
