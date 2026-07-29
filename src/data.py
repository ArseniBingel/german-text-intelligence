"""Loading the 10kGNAD corpus (Ten Thousand German News Articles).

Fetches the raw file once into a local cache, parses it, and applies minimal
cleaning. Downstream modules import load_articles/clean_articles rather than
reading the CSV themselves, so that every experiment in this repo starts from
byte-identical rows.

The raw file is a headerless, semicolon-separated CSV with two fields per row:
the topic label and the full article text. German news prose is full of
semicolons, which makes the separator ambiguous — pandas' default settings
raise a ParserError partway through the file rather than silently mangling it.

Cleaning here is deliberately conservative. The only rows removed are ones that
carry no signal at all. Lowercasing, stopword removal and stemming are
feature-extraction decisions: they belong to a later stage.

One property of the corpus shapes later work: the classes are imbalanced by
roughly three to one, from about 1,700 Panorama articles down to about 540
Kultur. Not severe, but enough that accuracy will flatter a model that has
learned very little.
"""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ARTICLES_URL = "https://raw.githubusercontent.com/tblock/10kGNAD/master/articles.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTICLES_PATH = DATA_DIR / "articles.csv"


def download_articles(url=ARTICLES_URL, dest=ARTICLES_PATH):
    """Ensure the raw corpus exists on disk. Returns the Path to it.

    Downloads only when the file is absent.

    Creates the parent directory if it does not exist yet, including any
    missing levels above it, and does not complain when it is already there.
    """
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        print(f"already there: {dest}")
        return dest

    urlretrieve(url,dest)

    return dest


def load_articles(path=ARTICLES_PATH):
    """Parse the raw 10kGNAD CSV into a DataFrame with columns 'label' and 'text'.

    The file has no header row — the first line is already data — and uses a
    semicolon separator. Semicolons also occur inside almost every article, so
    the text fields are single-quoted; the parser relies on sep=';' together
    with quotechar="'" to split the two columns correctly.

    Returns every row exactly as it appears in the file, with no filtering, so
    the effect of clean_articles can be measured as a before/after row count.
    """
    df = pd.read_csv(path, header=None, names=["label","text"], quotechar="'", sep=";")
    return df

def clean_articles(df:pd.DataFrame):
    """Drop rows that carry no signal. Returns a new DataFrame, same columns.

    Strips surrounding whitespace from both columns, removes rows whose text
    is empty once stripped, and removes rows whose text repeats one that
    appeared earlier, keeping the first occurrence.

    The caller's frame must be unchanged afterwards; several notebooks load
    once and clean repeatedly with different settings. The returned frame
    carries a fresh 0..n-1 index.
    """

    cleaned = df.copy()
    column_list = df.columns.to_list()
    for c in column_list:
        cleaned[c] = cleaned[c].str.strip()

    cleaned = cleaned[cleaned["text"] != ""]
    cleaned = cleaned.drop_duplicates(subset="text", keep="first")
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


if __name__ == "__main__":
    path = download_articles()
    assert path.exists(), "download_articles did not produce a file"
    print(f"file ok: {path.stat().st_size / 1e6:.1f} MB")

    raw = load_articles(path)
    assert isinstance(raw, pd.DataFrame)
    assert list(raw.columns) == ["label", "text"], raw.columns.tolist()
    assert len(raw) == 10_273, f"expected 10273 rows, got {len(raw)}"
    assert raw["label"].nunique() == 9, raw["label"].unique()
    assert raw["text"].dtype == object or pd.api.types.is_string_dtype(raw["text"])
    print(f"parsed ok: {len(raw)} rows, {raw['label'].nunique()} classes")

    before = len(raw)
    df = clean_articles(raw)
    print(df["label"].value_counts())
    assert len(raw) == before, "clean_articles mutated its input"
    assert list(df.columns) == ["label", "text"]
    assert df["text"].duplicated().sum() == 0, "duplicate texts remain"
    assert (df["text"].str.strip() == "").sum() == 0, "empty texts remain"
    assert list(df.index) == list(range(len(df))), "index was not reset"
    assert df["label"].nunique() == 9, "cleaning removed an entire class"
    print(f"cleaned ok: {before} -> {len(df)} rows")

    print(df["label"].value_counts())
    print(f"\n{df.iloc[0]['label']} | {df.iloc[0]['text'][:120]} ...")

    print("\ndata.py ran fine.")