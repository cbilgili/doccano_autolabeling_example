# coding: utf-8
import sqlite3
import json
from collections import defaultdict


def get_labels(conn, base_project=[1,3]):
    c = conn.cursor()
    out = defaultdict(dict)
    out2 = None
    for idx in base_project:
        c.execute("select * from server_label where project_id=?", (idx, ));
        labels = c.fetchall()
        for x in labels:
            out[x[1]][x[0]] = idx
    return out

def test_out(out, out2):
    assert(sorted(list(out.keys())) == sorted(out2))
    return True

def save(out, out2):
    with open("/doccano/labels.json", "w") as f:
        json.dump({"base":out, "ml":out2}, f)
    return True

def run(base_ids, ml_id):
    conn = sqlite3.connect("/doccano/app/db.sqlite3")
    out = get_labels(conn, base_ids)
    out2 = sorted(list(out.keys()))
    test_out(out, out2)
    save(out, out2)

if __name__ == "__main__":
    import sys
    base_ids = list(map(int, sys.argv[1].split(",")))
    run(base_ids, 2)
