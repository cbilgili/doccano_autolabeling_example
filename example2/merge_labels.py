import json
import sqlite3
import sys

def prepare(
        labelfile="/doccano/labels.json",
        dbfile="/doccano/app/db.sqlite3"
):
    with open(labelfile) as f:
        labdata = json.load(f)
    conn = sqlite3.connect(dbfile)
    return labdata, conn

def check_indices(q,indices):
    for i in indices:
        if i not in q:
            return False
    return True

def input_category(queue,i,out,out2):
    print(queue)
    try:
        tmp = input("{}>".format(i))
        if "exit" in tmp:
            return None,(None,None,None,None)
        tmp = tmp.replace(" ","")
        tmp = tmp.split(",")
        indices = list(map(int,tmp))
        if check_indices(queue, indices):
            out.append(indices)
            if len(indices) == 1:
                catname = queue[indices[0]]
                out2.append(catname)
                queue.pop(indices[0], None)
                i+=1
                return True,(queue,i,out,out2)
            for j in indices:
                queue.pop(j, None)
        else:
            return True,(queue,i,out,out2)
    except:
        return True,(queue,i,out,out2)
    i+=1
    return False,(queue,i,out,out2)
    

def input_category_name(out2):
    flag = True
    while(flag):
        catname = input("catname:")
        ok = None
        while(not(ok == 0 or ok == 1)):
            ok = int(input("Is '{}' OK? 1->yes, 0->no:".format(catname)))
        if ok == 1:
            flag = False
    ok = None
    flag = True
    out2.append(catname)
    return out2
        

def dialog(labels):
    queue = dict(enumerate(labels))
    out = []
    out2 = []
    i = 0
    flag = True
    while(queue):
        cont, (queue, i, out, out2) = input_category(queue, i, out, out2)
        if cont:
            continue
        if cont is None:
            return {}
        out2 = input_category_name(out2)
    result = {}
    for o1,o2 in zip(out,out2):
        for o in o1:
            result[o] = o2
    print(result)
    return result


if __name__ == "__main__":
    import json
    out = []
    labdata, conn = prepare()
    labels = labdata["ml"]
    result = dialog(labels)
    labels = dict(enumerate(labels))
    values = []
    i = 0
    for k,v in result.items():
        lids = list(labdata["base"][labels[k]].keys())
        for lid in lids:
            if v not in values:
                values.append(v)
                i += 1
            out.append((lid, (i,v)))
    print(out)
    with open("/doccano/labels_categorized.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
