"""Splits the data and builds the first baseline model for 10kGNAD.

Three jobs: split the data, build a TF-IDF + LogisticRegression pipeline, and
score that pipeline on the validation set.

The splitting lives here so every model in the project gets the exact same
rows. With a different split in another file, a difference between two models
could come from the split instead of the model, and there would be no way to
tell which.

The split is stratified because the classes are not balanced (see explore.py,
the biggest class is about 3x the smallest). Without stratify, small classes
like Kultur would get a different share in every split.

Only the validation set is used while trying things out. The test set gets used
once at the end in compare.py, after the model is already chosen. Checking the
test set after every change is a way of tuning on it, and then the final number
is no longer honest.

The vectorizer and the classifier go into one Pipeline instead of running one
after the other. The vocabulary and the IDF weights are learned from data, so
fitting the vectorizer before splitting puts information from the validation
rows into the features. That is leakage. A Pipeline fits the vectorizer on the
training part only, so the mistake cannot happen by accident.

TF-IDF is a real first attempt, not a throwaway. News topics mostly differ in
which words they use (Tor, Trainer vs Inflation, Notenbank), and word order
matters little for that. It also trains in a few seconds, which makes it a good
reference number for the slower models later.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data import load_articles, clean_articles
from sklearn.dummy import DummyClassifier





SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def make_splits(df, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED):
    """Splits the data into train, validation and test.

    Returns a dict with the keys 'train', 'val' and 'test'. Each value is a
    DataFrame with the same columns as df and a reset index. sklearn returns
    predictions as plain arrays, so position 0 of the array has to match row 0
    of the frame.

    train_test_split only makes two parts, so it has to run twice: first to take
    out the training rows, then to cut the rest into val and test.

    The second call counts from what is left over, not from
    the full data. Passing test_size straight through gives a test set that is too small.

    Stratify on the label in both calls.
    """

    val_size_calculated = val_size / (1-test_size)

    df_temp, df_test = train_test_split(df,test_size=test_size, stratify=df["label"], random_state=seed)
    df_train, df_val = train_test_split(df_temp, test_size=val_size_calculated, stratify=df_temp["label"], random_state=seed)



    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True)
    }


def make_vectorizer(min_df=3, max_df=0.6, ngram_range=(1, 1), max_features=None):
    """Builds the TF-IDF vectorizer. Returns it unfitted.

    Separate function so we can change the settings later without
    copying the defaults around.

    min_df drops tokens that appear in fewer than this many documents. explore.py
    counted about 189,000 different tokens, and most appear only once or twice:
    names, numbers, typos. A word seen three times cannot generalise, so
    dropping those makes the matrix much smaller at no cost in accuracy.

    max_df drops tokens that appear in more than this share of documents. IDF
    already handles those, so this is more of a safety net.

    sublinear_tf should be on. It uses 1 + log(tf) instead of the raw count. A
    word appearing 20 times in an article is not 20x more about the topic than
    a word appearing once, and documents here range from 3 to about 4800 words,
    so raw counts would be dominated by the long ones.
    """
    vec = TfidfVectorizer(min_df=min_df, lowercase=True, sublinear_tf=True, max_df=max_df, ngram_range=ngram_range, max_features=max_features)

    return vec


def make_baseline(vectorizer=None, C=1.0, class_weight=None, seed=SEED):
    """Puts the vectorizer and LogisticRegression into one Pipeline.

    Returns an unfitted Pipeline with the step names 'tfidf' and 'clf'. Builds
    a default vectorizer when none is passed in.

    LogisticRegression needs more than the default number of iterations here
    because there are so many features. A ConvergenceWarning means the
    optimiser stopped before it finished, so the coefficients are half done.
    That is different from the model simply being bad, so max_iter should be
    high enough that the warning disappears.

    class_weight is a parameter and not a default, on purpose. 'balanced'
    reweights the loss by class frequency. It usually helps macro-F1 a little
    and costs a little accuracy. That trade belongs in a validation test, not in
    a default, and the baseline should stay the simplest version.
    """
    if vectorizer is None:
        vectorizer = make_vectorizer()

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        random_state=seed,
        max_iter=1000
    )

    return Pipeline(
        [("tfidf", vectorizer),
         ("clf",clf)])

def evaluate(y_true,y_pred):
    """Scores one set of predictions against the true labels.

    Returns a dict with 'accuracy', 'macro_f1' and 'n'.

    Takes two label arrays rather than a model and a split, so the same
    function scores the TF-IDF pipeline, the embedding classifier and the
    majority baseline. Those three produce predictions in completely different
    ways, and none of that matters once the labels exist.

    Both scores are reported because they answer different questions. Accuracy
    is the share of articles predicted correctly. Macro-F1 averages the F1 of
    each class without caring about class size, so the 539 Kultur articles
    count as much as the 1678 Panorama ones.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true,y_pred, average="macro"),
        "n": len(y_true)
    }


def majority_baseline(train, val):
    """Scores a dummy model that always predicts the most common training label.

    Returns the same dict shape as evaluate().

    This is the floor. 0.85 accuracy sounds good but means nothing until the
    score for pure guessing is known. With nine classes of similar size that is
    around 0.16, and the gap between the two numbers is the honest version of
    "how much did the model actually learn".
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X=train["text"], y=train["label"])

    return evaluate(val["label"], dummy.predict(val["text"]))



def run_baseline(df=None, class_weight=None, C=1.0, seed=SEED, verbose=True):
    """Runs the whole thing: split, fit, score on validation, print a summary.

    Loads and cleans the data itself when df is None.

    Returns a dict with 'pipe', 'splits', 'metrics' and 'majority'.

    The splits come back too, so later models reuse these exact objects instead
    of building their own. Does not touch splits['test'].
    """
    if df is None:
        df = clean_articles(load_articles())

    splits_dict = make_splits(df)
    pipe = make_baseline(C=C,class_weight=class_weight,seed=seed)
    pipe.fit(splits_dict["train"]["text"],splits_dict["train"]["label"])
    metrics = evaluate(splits_dict["val"]["label"],
                       pipe.predict(splits_dict["val"]["text"]))
    majority = majority_baseline(splits_dict["train"],splits_dict["val"])

    if verbose:
        print(f"accuracy {metrics['accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f}")
        print(f"majority floor: accuracy {majority['accuracy']:.3f}")

    return {
        "pipe": pipe,
        "splits": splits_dict,
        "metrics": metrics,
        "majority": majority
    }



if __name__ == "__main__":
    df = clean_articles(load_articles())

    # split into train / val / test
    splits = make_splits(df)
    print("split sizes:", len(splits["train"]), len(splits["val"]), len(splits["test"]))
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(df)

    # train frame looks right: two columns, all 9 classes, index reset
    train = splits["train"]
    assert list(train.columns) == ["label", "text"]
    assert train["label"].nunique() == 9
    assert list(train.index) == list(range(len(train))), "index not reset"
    print("train has all 9 classes")

    # build and fit the baseline, score on validation
    result = run_baseline(df)
    metrics = result["metrics"]
    majority = result["majority"]

    print(f"majority: acc {majority['accuracy']:.4f}  macro-F1 {majority['macro_f1']:.4f}")
    print(f"baseline: acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")

    # the baseline must clearly beat "always guess the biggest class"
    assert metrics["accuracy"] > majority["accuracy"], "baseline is worse than guessing"
    assert metrics["accuracy"] > 0.80

    # vocabulary size after min_df / max_df trimming
    vocab = result["pipe"].named_steps["tfidf"].vocabulary_
    print("vocabulary size:", len(vocab))
    assert 40_000 < len(vocab) < 50_000

    # class_weight='balanced' usually trades a little accuracy for macro-F1
    balanced = run_baseline(df, class_weight="balanced", verbose=False)
    print(f"balanced: acc {balanced['metrics']['accuracy']:.4f}  "
          f"macro-F1 {balanced['metrics']['macro_f1']:.4f}")

    print("\nfeatures.py ran fine.")