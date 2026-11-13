"""
Soft-voting ensemble (Random Forest + XGBoost + MLP) with leave-one-fold-out
evaluation and a feature-group ablation.

Reads the same layout as the repo's feature_selection.py:
    Data2/Fold1/<file>.csv, Data2/Fold2/..., ...
    - one CSV per video, one row per frame, 9 columns (see FEATURES)
    - file name prefix decides the label:
        'alert'  -> 0
        'lowvig' -> 1   (low vigilant; merged into 'alert' with --classes 2)
        'drowsy' -> 2   (1 with --classes 2)
      The repo's original files (anything not 'drowsy' -> 0) also work with --classes 2.

Run from the project folder:
    venv311\\Scripts\\python.exe evaluate_ensemble.py --data Data3 --classes 3 --stride 5 [--window 20]

NOTE: "leave-one-fold-out" is only "unseen-subject" validation if no subject
appears in more than one fold. Check this before calling it that in the report.
"""
import argparse
import csv
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = ['MR', 'eye_closed', 'freq', 'perclo', 'eye_circ', 'pupil', 'eyebrow', 'moer', 'ear']
GROUPS = {
    'eye': ['eye_closed', 'freq', 'perclo', 'eye_circ', 'pupil', 'ear'],
    'mouth': ['MR', 'moer'],
    'eyebrow': ['eyebrow'],
}


def label_of(fname, n_classes):
    name = fname.lower()
    if name.startswith('drowsy'):
        return 1 if n_classes == 2 else 2
    if name.startswith('lowvig'):
        return 0 if n_classes == 2 else 1
    return 0


def load_fold(data_dir, fold, stride, n_classes, window=0):
    """Return X, y and a per-row video id for one fold.

    window > 1: each video is cut into non-overlapping segments of `window` rows
    (after striding) and each segment becomes one sample made of the mean and the
    standard deviation of every feature (18 values).
    """
    X, y, vid = [], [], []
    folder = os.path.join(data_dir, fold)
    for fname in sorted(os.listdir(folder)):
        label = label_of(fname, n_classes)
        rows = []
        with open(os.path.join(folder, fname), 'r') as f:
            for i, row in enumerate(csv.reader(f)):
                if not row or i % stride:
                    continue
                rows.append([float(v) for v in row])
        rows = np.nan_to_num(np.array(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if window > 1:
            for k in range(len(rows) // window):
                seg = rows[k * window:(k + 1) * window]
                X.append(np.concatenate([seg.mean(axis=0), seg.std(axis=0)]))
                y.append(label)
                vid.append(f'{fold}/{fname}')
        else:
            for r in rows:
                X.append(r)
                y.append(label)
                vid.append(f'{fold}/{fname}')
    return np.array(X), np.array(y), np.array(vid)


def make_models():
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        random_state=0, n_jobs=-1)
    mlp = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                      early_stopping=True, random_state=0))
    ens = VotingClassifier([('rf', rf), ('xgb', xgb), ('mlp', mlp)], voting='soft')
    return {'RandomForest': rf, 'XGBoost': xgb, 'MLP': mlp, 'Ensemble': ens}


def video_level_accuracy(y_true, y_pred, vids):
    """A video is correct if the majority of its frame predictions match its label."""
    hits = []
    for v in np.unique(vids):
        m = vids == v
        hits.append(int(np.bincount(y_pred[m]).argmax()) == int(y_true[m][0]))
    return float(np.mean(hits))


def evaluate(data, folds, cols, n_classes, model_names=None):
    """Leave-one-fold-out. Returns {model: metrics averaged over folds} + pooled confusion matrix."""
    scores, cm = {}, {}
    avg = 'binary' if n_classes == 2 else 'macro'
    for test_fold in folds:
        train_folds = [f for f in folds if f != test_fold]
        Xtr = np.vstack([data[f][0][:, cols] for f in train_folds])
        ytr = np.concatenate([data[f][1] for f in train_folds])
        Xte, yte, vte = data[test_fold][0][:, cols], data[test_fold][1], data[test_fold][2]
        for name, model in make_models().items():
            if model_names and name not in model_names:
                continue
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            s = scores.setdefault(name, {'acc': [], 'prec': [], 'rec': [], 'video_acc': []})
            s['acc'].append(accuracy_score(yte, pred))
            s['prec'].append(precision_score(yte, pred, average=avg, zero_division=0))
            s['rec'].append(recall_score(yte, pred, average=avg, zero_division=0))
            s['video_acc'].append(video_level_accuracy(yte, pred, vte))
            cm[name] = cm.get(name, 0) + confusion_matrix(yte, pred, labels=list(range(n_classes)))
        print(f'  finished test fold {test_fold}')
    return {n: {k: float(np.mean(v)) for k, v in s.items()} for n, s in scores.items()}, cm


def print_table(title, results):
    print(f'\n{title}')
    print(f'{"model":<14}{"acc":>8}{"prec":>8}{"recall":>8}{"video_acc":>11}')
    for name, r in results.items():
        print(f'{name:<14}{r["acc"]:>8.3f}{r["prec"]:>8.3f}{r["rec"]:>8.3f}{r["video_acc"]:>11.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='Data3')
    ap.add_argument('--classes', type=int, choices=[2, 3], default=3)
    ap.add_argument('--stride', type=int, default=5, help='keep every Nth frame (speed)')
    ap.add_argument('--window', type=int, default=0,
                    help='rows per segment after --stride (e.g. 20); 0 = classify single frames')
    args = ap.parse_args()

    folds = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    print(f'Folds found: {folds}')
    data = {f: load_fold(args.data, f, args.stride, args.classes, args.window) for f in folds}
    for f in folds:
        X, y, v = data[f]
        print(f'  {f}: {len(X)} rows, {len(np.unique(v))} videos, class counts {np.bincount(y, minlength=args.classes).tolist()}')

    names = FEATURES + [n + '_std' for n in FEATURES] if args.window > 1 else FEATURES
    all_idx = list(range(len(names)))
    base = lambda n: n[:-4] if n.endswith('_std') else n

    print('\n== Leave-one-fold-out, all 9 features, all models ==')
    res, cm = evaluate(data, folds, all_idx, args.classes)
    print_table('Results (mean over held-out folds; precision/recall are macro-averaged for 3 classes)', res)
    print('\nEnsemble confusion matrix (rows=true class, cols=predicted; classes in label order):')
    print(cm['Ensemble'])

    print('\n== Ablation (Ensemble only) ==')
    ablation = {'all features': res['Ensemble']}
    for g, gnames in GROUPS.items():
        keep = [i for i, n in enumerate(names) if base(n) not in gnames]
        ablation[f'without {g}'] = evaluate(data, folds, keep, args.classes, ['Ensemble'])[0]['Ensemble']
        only = [i for i, n in enumerate(names) if base(n) in gnames]
        ablation[f'{g} only'] = evaluate(data, folds, only, args.classes, ['Ensemble'])[0]['Ensemble']
    print_table('Feature-group ablation', ablation)

    out_name = f'results_{args.classes}class_window{args.window}.csv'
    with open(out_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'model', 'acc', 'prec', 'recall', 'video_acc'])
        for n, r in res.items():
            w.writerow(['all_features', n, r['acc'], r['prec'], r['rec'], r['video_acc']])
        for n, r in ablation.items():
            w.writerow([n, 'Ensemble', r['acc'], r['prec'], r['rec'], r['video_acc']])
    print(f'\nSaved {out_name}')


if __name__ == '__main__':
    main()