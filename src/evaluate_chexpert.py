# script used to get recall and precision of the modified chexpert-labeler over a 
# validation set of unseen reports from phase 1 and phase 2
import csv
import json
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from .global_paths import metadata_et_location

label_translation = {"fracture": "acute fracture & fracture",
              "acute fracture": "acute fracture & fracture",
              "airway wall thickening": "airway wall thickening",
              "atelectasis":"atelectasis",
              "consolidation":"consolidation",
              "enlarged cardiac silhouette":"enlarged cardiac silhouette",
              "enlarged hilum":"enlarged hilum",
              "groundglass opacity":"groundglass opacity",
              "hiatal hernia":"hiatal hernia",
              "high lung volume / emphysema":"high lung volume - emphysema & emphysema",
              "emphysema":"high lung volume - emphysema & emphysema",
              "interstitial lung disease":"interstitial lung disease & fibrosis",
              "fibrosis":"interstitial lung disease & fibrosis",
              "lung nodule or mass":"lung nodule or mass",
              "mass":"mass",
              "nodule":"nodule",
              "pleural abnormality":"pleural abnormality",
              "pleural effusion":"pleural effusion",
              "pleural thickening":"pleural thickening",
              "pneumothorax":"pneumothorax",
              "pulmonary edema":"pulmonary edema",
              "quality issue":"quality issue",
              "support devices":"support devices",
              "abnormal mediastinal contour":"wide mediastinum & abnormal mediastinum contour",
              "wide mediastinum":"wide mediastinum & abnormal mediastinum contour"}

def load_labels(path, label_indices, threshold=0):
    reval = []
    labels = []
    ids = []
    k = -1
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            observations = set()
            if i == 0:
                for j in label_indices:
                    t = row[j]
                    labels.append(label_translation[t.lower()])
                continue
            discard = row[2].strip().lower()
            if discard == 'true':
                continue
            for j, k in enumerate(label_indices):
                t = row[k]
                if float(t) > threshold:
                    observations.add(labels[j].lower())
            reval.append(observations)
            ids.append(row[0])
    label2id = {tag: i for i, tag in enumerate(labels)}
    return reval, label2id, ids

def load_pred(path, glabel2id):
    reval = []
    labels = []
    labels_ = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            pred_obs = set()
            
            if i == 0:
                columns_used = {}
                for j, t in enumerate(row[1:]):
                    if t.lower() in glabel2id:
                        index_ = glabel2id[t.lower()]
                        labels_.append(t.lower())
                        labels.append(t.lower())
                        columns_used[j] = len(labels) - 1
            else:
                for j, t in enumerate(row[1:]):
                    if j in columns_used:
                        if len(t) > 0 and (float(t) == 1 or float(t) == -1):
                            pred_obs.add(labels_[columns_used[j]].lower())
                reval.append(pred_obs)
    label2id = {tag: i for i, tag in enumerate(labels)}
    return reval, label2id

def evaluate(gold_labels, pred_labels, label2id):
    recalls = []
    precs = []
    labels = []
    for label, id_ in label2id.items():
        gold_vec = []
        pred_vec = []
        for gold, pred in zip(gold_labels, pred_labels):
            t = 0
            if label in gold:
                t = 1
            gold_vec.append(t)
            t = 0
            if label in pred:
                t = 1
            pred_vec.append(t)
        score = recall_score(gold_vec, pred_vec)
        recalls.append(score)
        score = precision_score(gold_vec, pred_vec)
        precs.append(score)
        labels.append(label)

    print(','.join(labels))
    print(','.join([str(t) for t in recalls]))
    print(','.join([str(t) for t in precs]))

def load_text(path):
    reval = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            else:
                reval.append(row[0])
    return reval

def analyze(path, ids, text, gold_labels, pred_labels):
    data = []
    for id_, sent, glabel, plabel in zip(ids, text, gold_labels, pred_labels):
        ins = {'id': id_, 'text': sent}
        ins['gold labels'] = sorted(list(glabel))
        ins['detected labels'] = sorted(list(plabel))
        data.append(ins)
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_ids(path):
    reval = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            reval.add(s)
    return reval

if __name__ == '__main__':
    phase = 2
    path = f'{metadata_et_location}/metadata_phase_{phase}.csv'
    label_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22]

    gold_labels, glabel2id, ids = load_labels(path, label_indices)
    
    # loading the cases that were used to define the modifications to the chexpert-labeler 
    # (20% of the original data)
    checked_ids = load_ids('ids_radiologist_validation_phases_1_and_2.txt')
    
    path = f'labeled_reports_{phase}.csv'
    pred_labels, plabel2id = load_pred(path,glabel2id)
    text = load_text(path)

    print(glabel2id)
    print(plabel2id)
    assert glabel2id.keys() == plabel2id.keys()
    assert len(gold_labels) == len(pred_labels)
    assert len(text) == len(gold_labels)

    # Filter the checked ids- removing examples that were used to define new
    # rules, and only leaving unseen examples from phase 1 and 2 for calculating validation metrics
    fgold_labels = []
    fpred_labels = []
    fids = []
    ftext = []
    for a, b, c, _id in zip(gold_labels, pred_labels, text, ids):
        if _id not in checked_ids:
            fgold_labels.append(a)
            fpred_labels.append(b)
            ftext.append(c)
            fids.append(_id)
    assert len(gold_labels) == len(pred_labels)
    assert len(ftext) == len(fgold_labels)
    assert len(fids) == len(fgold_labels)

    evaluate(fgold_labels, fpred_labels, glabel2id)
    path = f'phase_{phase}_result.json'
    analyze(path, fids, ftext, fgold_labels, fpred_labels)
