"""Exploratory analysis of the 10kGNAD corpus.

Answers the questions that decide how the baseline in features.py gets built:
how skewed are the classes, how long are the documents, and does the raw
vocabulary carry any class signal at all.

Nothing here modifies data. Every function takes the cleaned frame from
data.py and returns a summary, so this module can be re-run freely and its
output pasted straight into the README.

Two findings from this file drive later decisions. First, the classes are
imbalanced by roughly three to one, which is why eval.py reports macro-F1
instead of accuracy: a model that ignored the two smallest classes entirely
would still look respectable on accuracy alone. Second, raw token frequency is
dominated almost completely by German function words — die, der, und, in. Every
class has the same top twenty. That is the whole motivation for the IDF
weighting in the next block: a word that appears everywhere cannot separate
anything, so it has to be down-weighted before the counts become useful.

Document length also differs systematically by class, and that one deserves
thought rather than celebration. If Inland articles run longer than Web
articles, a classifier can exploit length as a shortcut. Whether that counts as
real signal or as an artefact of how this particular newsroom works is a
judgement call, and the answer would be different for maintenance reports.
"""
import re
from collections import Counter

import pandas as pd

from data import load_articles, clean_articles

TOKEN_PATTERN = r"\w+"


def class_distribution(df: pd.DataFrame):
    """Absolute and relative frequency of each label.

    Returns a DataFrame indexed by label with columns 'n' and 'share',
    sorted from most to least frequent. 'share' is a proportion, not a
    percentage, and sums to 1.
    """
    counts = df["label"].value_counts()
    share = df["label"].value_counts(normalize=True)

    return pd.DataFrame({"n": counts,
                        "share": share})




def imbalance_ratio(dist:pd.DataFrame):
    """How many times larger the biggest class is than the smallest.

    Takes the frame returned by class_distribution. Returns a float.

    Anything near 1.0 means balanced classes and accuracy is a defensible
    metric. The further above 1.0, the more accuracy rewards a model for
    simply ignoring the rare classes.
    """
    max_share = dist["share"].max()
    min_share = dist["share"].min()
    return max_share / min_share


def add_length_features(df):
    """Return a copy of df with 'n_chars' and 'n_words' added.

    n_words splits on whitespace rather than on the token regex, because the
    question here is document length as a human would count it, not vocabulary
    size.

    The input frame must be unchanged afterwards.
    """
    out = df.copy()
    out["n_chars"] = out["text"].str.len()
    out["n_words"] = out["text"].str.split().str.len()

    return out


def length_by_class(df:pd.DataFrame):
    """Per-class document length statistics.

    Takes a frame that already carries the length columns. Returns a DataFrame
    indexed by label with columns 'n', 'median_words', 'mean_words' and
    'p90_words', sorted by median ascending.

    Median and mean are both reported deliberately. The corpus contains a small
    number of very long articles, and where the mean sits far above the median
    the class has a long tail rather than a genuinely different typical length.
    """

    stats = df.groupby("label")["n_words"].agg(
        n="size",
        median_words="median",
        mean_words="mean",
        p90_words=lambda s: s.quantile(0.90)
    )

    return stats.sort_values("median_words")

def top_tokens(texts, n=20):
    """The n most frequent tokens across an iterable of strings.

    Returns a list of (token, count) tuples, most frequent first.

    Lowercases before counting, so that 'Die' at the start of a sentence and
    'die' inside one are not treated as two separate words. Tokenises with
    TOKEN_PATTERN, which is Unicode-aware — an ASCII-only pattern would quietly
    cut German words at the umlaut and hand you a corrupted vocabulary that
    still looks plausible.

    Accumulates into a single Counter while iterating rather than building one
    list of all tokens first; the corpus is around three million tokens.
    """
    counter = Counter()

    for entry in texts:
        counter.update(re.findall(TOKEN_PATTERN,entry.lower()))

    return counter.most_common(n)

def count_tokens_by_class(df):
    """Tokenise the corpus once, tallying separately per label.

    Returns a dict mapping each label to a Counter of its token frequencies.

    One pass over the text column, dispatching each documents tokens into the
    counter for its own label. Everything else in this module that needs
    frequencies is derived from this result rather than re-reading the text.
    """
    per_class = {}
    for label, text in zip(df["label"], df["text"]):
        if label not in per_class:
            per_class[label] = Counter()
        per_class[label].update(re.findall(TOKEN_PATTERN, text.lower()))
    return per_class


def combine_counters(per_class):
    """Merge the per-class counters into one corpus-wide Counter.

    Takes the dict returned by count_tokens_by_class. Returns a Counter.

    The corpus totals are the sum of the class totals, so no further pass over
    the documents is needed. Prefer accumulating into a single Counter over
    repeatedly adding counters together: the + operator builds a new object
    each time and discards the previous one, which turns nine merges into
    nine allocations of a 189,000-key dictionary.
    """
    corpus = Counter()
    for counter in per_class.values():
        corpus.update(counter)         
    return corpus


def distinctive_tokens(class_counter, corpus_counter, n=15, min_count=50):
    """Tokens that are over-represented in one class, relative to the corpus.

    Returns a list of (token, lift) tuples sorted by lift descending, where
    lift is the token's rate inside the class divided by its rate across the
    whole corpus.

    Ranking by raw frequency inside the class would return the same German
    function words for all nine labels and tell you nothing. The ratio asks
    the more useful question: given how common this word is in general, is it
    unusually common here.

    min_count is a floor on the token's count inside the class. A token
    occurring twice, both times in this class, has an enormous lift and no
    evidential value whatsoever; the floor is what keeps the output readable.
    """
    n_label = sum(class_counter.values())
    n_corpus = sum(corpus_counter.values())

    result = []
    for token, count_in_group in class_counter.items():
        if count_in_group < min_count:
            continue
        rate_in_group = count_in_group / n_label
        rate_overall = corpus_counter[token] / n_corpus
        result.append((token, rate_in_group / rate_overall))

    result.sort(key=lambda x: x[1], reverse=True)
    return result[:n]


def explore(df=None, n_tokens=20):
    if df is None:
        df = load_articles()
        df = clean_articles()

    class_dis = class_distribution(df)
    ratio = imbalance_ratio(class_dis)
    df = add_length_features(df)
    stats = length_by_class(df)
    corpus_top = top_tokens(df["text"], n_tokens)

    class_counter = count_tokens_by_class(df)
    corpus_counter = combine_counters(class_counter)

    distinctive = {
        label: distinctive_tokens(class_counter[label], corpus_counter, n_tokens)
        for label in class_counter
    }

    return {
        "distribution": class_dis,
        "imbalance": ratio,
        "lengths": stats,
        "corpus_tokens": corpus_counter,
        "distinctive": distinctive,
    }


if __name__ == "__main__":
    df = clean_articles(load_articles())
    print("rows after cleaning:", len(df))
    assert len(df) == 10_271

    # class distribution: 9 labels, shares add to 1
    dist = class_distribution(df)
    print("\n", dist)
    assert len(dist) == 9
    assert dist["n"].sum() == len(df)              # every row counted once
    assert abs(dist["share"].sum() - 1.0) < 1e-9   # shares sum to 100%

    ratio = imbalance_ratio(dist)
    print("\nimbalance ratio: %.2fx" % ratio)
    assert ratio > 1.0                             # some class is bigger than another

    # length features: added without changing the original df
    n_cols_before = len(df.columns)
    with_len = add_length_features(df)
    assert len(df.columns) == n_cols_before        # original untouched
    assert "n_words" in with_len.columns
    assert (with_len["n_chars"] >= with_len["n_words"]).all()   # chars >= words always

    lengths = length_by_class(with_len)
    print("\n", lengths)
    assert len(lengths) == 9

    # most frequent tokens
    tokens = top_tokens(with_len["text"], n=20)
    print("\ntop tokens:", [t for t, _ in tokens])
    assert len(tokens) == 20
    assert "für" in [t for t, _ in tokens], "umlaut lost — check the token pattern"

    # distinctive tokens per class: should be over-represented (lift > 1)
    class_counter = count_tokens_by_class(df)
    corpus_counter = combine_counters(class_counter)
    for label in ["Sport", "Kultur", "Wirtschaft"]:
        marked = distinctive_tokens(class_counter[label], corpus_counter, n=10)
        print(f"{label:12s}", ", ".join(t for t, _ in marked))
        assert all(lift > 1.0 for _, lift in marked)

    report = explore(df)
    assert set(report) == {"distribution", "imbalance", "lengths",
                           "corpus_tokens", "distinctive"}

    print("\nexplore.py ran fine.")