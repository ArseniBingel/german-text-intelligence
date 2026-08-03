"""Detailed evaluation of a fitted model on the validation split.

features.py gives one accuracy and one macro-F1. Those two numbers say how well
the model does but nothing about where it fails, and the where is the part that
suggests what to fix.

This file breaks the score down per class, builds the confusion matrix, lists
the label pairs that get mixed up most, and pulls out the actual articles that
were classified wrongly so they can be read.

Reading the wrong ones matters. A confusion matrix says Etat gets mistaken for
Panorama; only the articles themselves say whether that is a model problem, a
messy category boundary, or a labelling mistake in the corpus. Those three
cases need completely different fixes.

Cross-validation is in here too, for a different reason. The validation score
comes from one split of 1541 articles, so part of it is luck. Running five
folds on the training data gives a mean and a std and the std says how
big a difference between two models has to be before it means anything. A model
that scores 0.005 higher than another is not better if the fold-to-fold spread
is 0.012.

Note that the cross-validation mean comes out lower than the single validation
score. That is expected: each fold trains on four fifths of the training data
instead of all of it. The two numbers answer different questions and should not
be compared directly.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data import load_articles, clean_articles
from features import SEED, run_baseline


def per_class_report(y_true, y_pred, labels=None):
    """Precision, recall, F1 and support for every class.

    Returns a DataFrame with the columns 'label', 'precision', 'recall', 'f1'
    and 'support', sorted by f1 ascending so the weakest class is on top.

    Uses the sorted unique labels from y_true when labels is None.

    precision_recall_fscore_support returns four arrays in label order, so the
    labels list has to be passed in and reused when building the frame.
    Otherwise the numbers silently line up with the wrong classes.

    Precision and recall pull apart in a way the F1 alone hides. High precision
    with low recall means the model is too careful about a class: whatever it
    labels is right, but it misses most of them. The opposite means the class
    is being used as a catch-all. The fix is different in each case.
    """
    if labels is None:
        labels = sorted(pd.unique(y_true))

    precision, recall, f1, support = precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)

    return pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    }).sort_values("f1").reset_index(drop=True)


def confusion(y_true, y_pred, labels=None):
    """Confusion matrix as a labelled DataFrame.

    Rows are the true label, columns are the prediction. Index and columns are
    named 'true' and 'predicted' so the orientation is visible when printing.

    confusion_matrix returns a plain array with no labels attached, so the same
    labels list has to go into both index and columns.
    """
    if labels is None:
        labels = sorted(pd.unique(y_true))

    cm = confusion_matrix(y_true,y_pred,labels=labels)

    table = pd.DataFrame(cm,index=labels,columns=labels)
    table.index.name = "true"
    table.columns.name = "predicted"

    return table


def top_confusions(cm:pd.DataFrame, n=10):
    """The label pairs that get mixed up most often.

    Takes the DataFrame from confusion(). Returns a list of
    (true_label, predicted_label, count) tuples sorted by count descending.

    The diagonal is the correct predictions and has to be skipped, otherwise
    the list is just the biggest classes.
    """
    result = []

    for true_label in cm.index:
        for pred_label in cm.columns:
            if true_label == pred_label:
                continue
            result.append((true_label,pred_label,int(cm.loc[true_label,pred_label])))

    result.sort(key=lambda row: row[2], reverse=True)

    return result[:n]


def misclassified(split:pd.DataFrame, y_pred, true_label=None, pred_label=None, n=10, chars=300):
    """The articles that were classified wrongly, for reading by hand.

    Returns a DataFrame with the columns 'true', 'predicted' and 'snippet',
    where snippet is the first `chars` characters of the text.

    true_label and pred_label filter to one specific confusion when given, so a
    single cell of the confusion matrix can be inspected. With both None, all
    wrong rows are eligible.

    Returns at most n rows.
    """
    out = pd.DataFrame({
        "true": split["label"].to_numpy(),
        "predicted": y_pred,
        "snippet": split["text"].str[:chars].to_numpy()
    })

    out = out[out["true"] != out["predicted"]]

    if true_label is not None:
        out = out[out["true"] == true_label]
    if pred_label is not None:
        out = out[out["predicted"] == pred_label]

    return out.head(n) 


def cv_macro_f1(pipe, train, n_folds=5, seed=SEED):
    """Cross-validated macro-F1 on the training split.

    Returns a dict with 'scores' (the array of per-fold scores), 'mean' and
    'std'.

    The pipeline has to go in unfitted. cross_val_score clones it and refits it
    inside every fold, which is what keeps the vectorizer from ever seeing the
    fold it is being scored on.

    Shuffles the folds with a fixed seed. Without shuffling, the folds follow
    the row order of the data, and any leftover ordering in the corpus turns
    into a difference between folds that has nothing to do with the model.

    The std is the useful part. It is the size of the difference that could
    appear between two identical models just from how the rows were divided.
    """
    folds = StratifiedKFold(n_splits=n_folds,random_state=seed,shuffle=True)
    scores = cross_val_score(pipe,X=train["text"],y=train["label"],cv=folds, scoring="f1_macro")

    return {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std()
    }


def run_eval(result=None, n_examples=5):
    """Full validation-set evaluation of a fitted baseline: metrics plus a
    hand-readable error sample.

    Takes the dict from run_baseline and reuses its fitted pipeline and splits,
    builds a fresh baseline itself when result is None.

    Steps, in order:
      1. Predict the validation labels with the fitted pipeline.
      2. per_class_report  -> precision/recall/F1/support per class, weakest first.
      3. confusion         -> the labelled 9x9 true-vs-predicted matrix.
      4. top_confusions    -> the n_examples label pairs mixed up most often.
      5. cv_macro_f1       -> cross-validated macro-F1 on the TRAINING split,
                              so the val score comes with an error bar.
      6. misclassified is called twice: once unfiltered to collect every error
         (returned as 'wrong' for further digging), and once filtered to the
         single worst confusion pair, whose articles are printed so they can be
         read by hand.

    The printing in step 6 is the point of the function. The metrics say which
    class is weak and what it gets confused with; only reading the raw articles
    behind that top confusion tells you whether the two categories genuinely
    overlap or the model is simply weak on them.

    Returns a dict with keys 'report', 'cm', 'confusions', 'cv' and 'wrong'.
    """
    if result is None:
        df = clean_articles(load_articles())
        result = run_baseline(df)

    pipe = result["pipe"]
    val_df = result["splits"]["val"]
    y_true = val_df["label"]
    y_pred = pipe.predict(val_df["text"])
    report = per_class_report(y_true,y_pred)
    cm = confusion(y_true,y_pred)
    confusions = top_confusions(cm,n=n_examples)
    cv =  cv_macro_f1(pipe,result["splits"]["train"])

    wrong = misclassified(val_df, y_pred, n=None)
    if confusions:
        true_label, pred_label, count = confusions[0]
        examples = misclassified(val_df, y_pred,
                                    true_label=true_label, pred_label=pred_label,
                                    n=n_examples)
        print(f"\n{true_label} misread as {pred_label} ({count} times) — {n_examples} examples:\n")
        for _, row in examples.iterrows():
            print(f"  [{row['true']} → {row['predicted']}] {row['snippet']}\n")
   

    return {
        "report": report,
        "cm": cm,
        "confusions": confusions,
        "cv": cv,
        "wrong": wrong,
        "y_pred": y_pred
    }
    


if __name__ == "__main__":
    df = clean_articles(load_articles())
    result = run_baseline(df, verbose=False)

    val = result["splits"]["val"]
    y_true = val["label"]
    y_pred = result["pipe"].predict(val["text"])

    report = per_class_report(y_true, y_pred)
    assert len(report) == 9
    assert list(report.columns) == ["label", "precision", "recall", "f1", "support"]
    assert report["support"].sum() == 1541
    assert report.iloc[0]["f1"] <= report.iloc[-1]["f1"], "not sorted by f1"
    assert report.iloc[0]["label"] == "Etat", report.iloc[0]["label"]
    assert report.iloc[-1]["label"] == "Sport", report.iloc[-1]["label"]
    print(report.round(3).to_string(index=False))

    worst = report.iloc[0]
    assert worst["precision"] > 0.9, worst["precision"]
    assert worst["recall"] < 0.7, worst["recall"]
    print(f"\nworst class {worst['label']}: precision {worst['precision']:.2f} "
          f"but recall {worst['recall']:.2f}")

    cm = confusion(y_true, y_pred)
    assert cm.shape == (9, 9)
    assert cm.values.sum() == 1541
    correct = np.trace(cm.values)
    assert correct > 1200, correct
    print("correct predictions:", correct, "wrong:", 1541 - correct)

    pairs = top_confusions(cm, n=5)
    assert len(pairs) == 5
    assert len(pairs[0]) == 3
    assert pairs[0][0] != pairs[0][1], "diagonal not skipped"
    assert pairs[0][2] >= pairs[-1][2], "not sorted by count"
    print("\ntop confusions:")
    for true_label, pred_label, count in pairs:
        print(f"  {true_label} -> {pred_label}: {count}")

    wrong = misclassified(val, y_pred, n=5)
    assert len(wrong) == 5
    assert list(wrong.columns) == ["true", "predicted", "snippet"]
    assert (wrong["true"] != wrong["predicted"]).all(), "these are not errors"

    etat = misclassified(val, y_pred, true_label="Etat", pred_label="Panorama", n=3)
    assert len(etat) <= 3
    assert (etat["true"] == "Etat").all()
    print("\nEtat articles predicted as Panorama:")
    for _, row in etat.iterrows():
        print(f"  {row['snippet'][:160]} ...")

    cv = cv_macro_f1(result["pipe"], result["splits"]["train"])
    assert len(cv["scores"]) == 5
    assert 0.78 < cv["mean"] < 0.85, cv["mean"]
    assert cv["std"] < 0.05, cv["std"]
    print(f"\ncv macro-F1: {cv['mean']:.4f} +/- {cv['std']:.4f}")
    print("per fold:", cv["scores"].round(4))

    print("\neval.py ran fine.")