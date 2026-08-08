"""Sentence embeddings as an alternative to TF-IDF.

TF-IDF treats every word as its own column, so two articles that say the same
thing in different words look completely unrelated. An embedding model maps a
whole text to one vector of fixed length, and texts with similar meaning land
close together even when they share no words.

The plan here is embeddings plus a LogisticRegression head, not fine-tuning.
Fine-tuning updates the weights of the transformer itself and needs a GPU and
hours; this keeps the encoder frozen, turns each article into one vector once,
and trains a cheap classifier on top. On roughly 7000 training documents that
is the sensible trade, and "fine-tuning was considered and rejected because the
cost did not fit the data size" is a real answer, not an excuse.

The encoder is not trained on this corpus at all, so unlike TF-IDF there is no
vocabulary to leak. The whole dataset can be encoded in one go before splitting,
which is also why the cache is worth having: encoding 10,000 articles on CPU
takes several minutes, and nothing about the result changes between runs.

The important limitation is the sequence length. This model truncates at 128
tokens. explore.py measured a median article of about 298 words, so most of
every article never reaches the model - the vector describes the opening
paragraph only. That is the main reason to expect this model to lose against
TF-IDF here, which sees every word of every document. It is a real result and
worth reporting rather than hiding.

The model is multilingual and was not built specifically for German. A
German-only encoder like deepset/gbert-base would likely do better, at the cost
of a larger download and slower encoding. Trying it is a later experiment.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from data import load_articles, clean_articles
from features import SEED, evaluate, make_splits, run_baseline

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_PATH = CACHE_DIR / "embeddings.npy"


def load_encoder(model_name=MODEL_NAME):
    """Load the sentence transformer. Returns the model.

    The import of SentenceTransformer belongs inside this function, not at the
    top of the file. It pulls in torch and takes several seconds, and the other
    functions here should stay importable without paying that cost.

    Downloads about 470 MB on the first call and caches it in the user's home
    directory, so later calls are fast.
    """
    from sentence_transformers import SentenceTransformer
    
    return SentenceTransformer(model_name,cache_folder=CACHE_DIR)



def embed_texts(model, texts, batch_size=32, show_progress=True):
    """Encode a list of strings into vectors.

    Returns a float32 array of shape (len(texts), embedding_dim).

    normalize_embeddings should be on. It scales every vector to length 1,
    which makes the dot product equal to the cosine similarity and gives
    LogisticRegression features on a consistent scale. Without it, longer texts
    tend to produce larger vectors and that length difference becomes a feature
    the classifier can pick up on for no good reason.

    encode() already batches internally, so no manual loop is needed. Passing
    the whole list at once is fine and faster than calling it per document.
    """
    vectors = model.encode(texts,normalize_embeddings=True,batch_size=batch_size,show_progress_bar=show_progress)

    return vectors


def embed_corpus(df, path=EMBED_PATH, model_name=MODEL_NAME):
    """Encode every article in df, using the cached file when it exists.

    Returns an array with one row per row of df, in the same order.

    Encoding takes several minutes on CPU and the result never changes, so it
    gets saved to disk as .npy and reloaded on later runs.

    The cache is keyed only by filename, which is a weakness: changing the
    model or the cleaning rules while an old file is lying around silently
    loads the wrong vectors. Guard against it by checking that the row count of
    the cached array matches len(df), and delete the file by hand after
    changing the model.
    """
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(df):
            return cached

    encoder = load_encoder(model_name)
    embeddings = embed_texts(encoder,df["text"].tolist())

    path.parent.mkdir(exist_ok=True,parents=True)

    np.save(path,embeddings)

    return embeddings


def split_embeddings(embeddings, df, seed=SEED):
    """Split the embedding array the same way make_splits splits the frame.

    Returns a dict with keys 'train', 'val' and 'test', each a
    (X, y) tuple of an embedding array and a label Series.

    The splits have to match the ones in features.py exactly, or the comparison
    between the two models is meaningless. Rather than splitting the array
    directly, add a column of positions to a copy of df, run make_splits on
    that, and use the positions in each part to index into the array.

    Do not re-split with a fresh random_state and hope it lines up. The seed
    alone is not enough of a guarantee if anything about the frame changed.
    """
    df_pos = df.copy()
    df_pos["_pos"] = range(len(df_pos))

    splits = make_splits(df=df_pos,seed=seed)

    out = {}
    for name in ["train","val","test"]:
        part = splits[name]
        idx = part["_pos"].to_numpy()
        out[name] = (embeddings[idx],part["label"])
    return out

def run_embedding_model(df=None, C=1.0, seed=SEED, verbose=True):
    """Encode the corpus, fit LogisticRegression on the vectors, score on validation.

    Loads and cleans the data itself when df is None.

    Returns a dict with 'clf', 'splits', 'metrics' and 'embeddings'.

    No Pipeline here, unlike features.py. The encoder never sees the labels and
    is not fitted on this data at all, so there is no transform that could leak
    validation information into the features.

    Does not touch the test split.
    """
    if df is None:
        df = clean_articles(load_articles())

    embeddings = embed_corpus(df,path=EMBED_PATH,model_name=MODEL_NAME)
    split = split_embeddings(embeddings=embeddings,df=df,seed=seed)
    X_train,y_train = split["train"]
    clf = LogisticRegression(C=C,max_iter=1000,random_state=seed)
    clf.fit(X=X_train,y=y_train)

    X_val,y_val = split["val"]

    metrics = evaluate(y_val,clf.predict(X_val))

    return {
        "clf": clf,
        "splits": split,
        "metrics": metrics,
        "embeddings": embeddings
    }

if __name__ == "__main__":
    model = load_encoder()
    dim = model.get_embedding_dimension()
    assert dim == 384, dim
    print("model loaded, embedding dim:", dim)
    print("max sequence length:", model.max_seq_length)

    sample = embed_texts(model, ["Das Spiel endete 2:1.", "Die Inflation stieg."],
                         show_progress=False)
    assert sample.shape == (2, 384), sample.shape
    assert abs(np.linalg.norm(sample[0]) - 1.0) < 0.01, "vectors not normalized"
    print("encoding ok, shape:", sample.shape, sample.dtype)

    df = clean_articles(load_articles())
    embeddings = embed_corpus(df)
    assert embeddings.shape == (len(df), 384), embeddings.shape
    assert not np.isnan(embeddings).any(), "NaN in the embeddings"

    again = embed_corpus(df)
    assert np.array_equal(embeddings, again), "cache gave different vectors"
    print("cache ok")

    parts = split_embeddings(embeddings, df)
    assert len(parts) == 3
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    assert X_train.shape == (7189, 384), X_train.shape
    assert X_val.shape == (1541, 384), X_val.shape
    assert len(y_train) == 7189
    print("splits ok:", X_train.shape, X_val.shape)

    result = run_embedding_model(df, verbose=False)
    metrics = result["metrics"]
    assert metrics["n"] == 1541
    assert metrics["accuracy"] > 0.60, metrics["accuracy"]
    assert metrics["macro_f1"] > 0.55, metrics["macro_f1"]

    tfidf = run_baseline(df, verbose=False)["metrics"]
    print(f"tf-idf:     acc {tfidf['accuracy']:.4f}  macro-F1 {tfidf['macro_f1']:.4f}")
    print(f"embeddings: acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")

    print("embeddings.py ran fine.")