# coding: utf-8
import sqlite3
import json

def connect(dbpath="/doccano/app/db.sqlite3"):
    return sqlite3.connect(dbpath)


def get_labels(conn, base_project=[1,3], ml_project=2):
    c = conn.cursor()
    out = {}
    out2 = None
    for idx in base_project:
        c.execute("select * from server_label where project_id=?", (idx, ));
        labels = c.fetchall()
        labels = {x[0]:x[1] for x in labels}
        out[idx] = labels
    c.execute("select * from server_label where project_id=?", (ml_project, ));
    labels = c.fetchall()
    labels = {x[0]:x[1] for x in labels}
    out2 = labels
    return out, out2


def test_out(out, out2):
    base = None
    for k,v in out.items():
        if base is None:
            base = sorted(list(v.values()))
        else:
            assert(base == sorted(list(v.values())))
    assert(base == sorted(list(out2.values())))
    return True


def save(out, out2):
    with open("/doccano/labels.json", "w") as f:
        json.dump({"base":out, "ml":out2}, f)
    return True

def run(base_ids):
    conn = connect()
    out, out2 = get_labels(conn, base_ids)
    test_out(out, out2)
    save(out,out2)

if __name__ == "__main__":
    import sys
    base_ids = list(map(int, sys.argv[1].split(",")))
    run(base_ids)
