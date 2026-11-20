import argparse, os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def model_fn(model_dir):                       # used by the inference container
    return joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST"))
    args = p.parse_args()

    tr = pd.read_csv(os.path.join(args.train, "train.csv"))
    te = pd.read_csv(os.path.join(args.test, "test.csv"))
    X_tr, y_tr = tr.drop("target", axis=1), tr["target"]
    X_te, y_te = te.drop("target", axis=1), te["target"]

    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    print(f"test_accuracy={accuracy_score(y_te, pred):.4f}")
    print(f"test_precision={precision_score(y_te, pred):.4f}")
    print(f"test_recall={recall_score(y_te, pred):.4f}")
    print(f"test_f1={f1_score(y_te, pred):.4f}")
    print(f"test_auc={roc_auc_score(y_te, prob):.4f}")
    print(f"test_recall_malignant={recall_score(y_te, pred, pos_label=0):.4f}")
    print(f"confusion_tn={tn} confusion_fp={fp} confusion_fn={fn} confusion_tp={tp}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
