"""Rule-based extraction of fields from German maintenance reports.

Reads the corpus from reports.py, pulls each field out with a regex, validates
the result against a Pydantic schema, and scores it against the ground truth.

The score is broken down per field and per report style, because a single
overall number hides where the rules fail. Regex is expected to do well on the
'complete' style and badly on 'freetext'.

The same scorer is reused for the LLM extractor later, so both approaches are
measured on identical documents with identical rules.
"""
import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from collections import defaultdict

from reports import load_reports


FIELDS = ["report_date", "asset_id", "component", "error_code",
          "severity", "downtime_minutes", "technician"]

MONTHS = {"Januar": 1, "Februar": 2, "März": 3, "April": 4, "Mai": 5,
          "Juni": 6, "Juli": 7, "August": 8, "September": 9,
          "Oktober": 10, "November": 11, "Dezember": 12}

SEVERITIES = ["niedrig", "mittel", "hoch", "kritisch"]


class ReportFields(BaseModel):
    """Schema for one extracted report.

    All fields are optional, because a report may genuinely not contain them.
    Validation checks the format of what was found, not whether it was found.

    asset_id must match RM-1234-A. error_code must match E-1234. severity must
    be one of SEVERITIES in lower case. downtime_minutes must be between 0 and
    10080, which is one week - a larger value means the unit was misread.
    """
    report_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    asset_id: Optional[str] = Field(default=None, pattern=r"^RM-\d{4}-[A-D]$")
    component: Optional[str] = None
    error_code: Optional[str] = Field(default=None, pattern=r"^E-\d{4}$")
    severity: Optional[Literal["niedrig", "mittel", "hoch", "kritisch"]] = None
    downtime_minutes: Optional[int] = Field(default=None, ge=0, le=10080)
    technician: Optional[str] = None


def find(pattern, text):
    """Searches for a pattern and returns the 'value' group.

    Args:
        pattern: regex string containing a named group 'value'.
        text: the text to search.

    Returns:
        str or None if the pattern does not match.
    """
    match = re.search(pattern=pattern,string=text)

    return match.group("value") if match else None


def parse_date(text):
    """Extracts a date and returns it as 'YYYY-MM-DD'.

    Args:
        text: report text.

    Returns:
        str or None.

    Handles three formats: 2024-03-14, 14.03.2024, and 14. März 2024. Must try
    them in that order, otherwise a looser pattern matches part of a string the
    stricter one would have handled.
    """
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)         # ISO
    if match:
        return match.group(0)

    match = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)   # dd.mm.yyyy
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    match = re.search(r"(\d{1,2})\.\s*(\w+)\s+(\d{4})", text)   # 14. März 2024
    if match:
        day, month, year = match.groups()
        if month in MONTHS:
            return f"{year}-{MONTHS[month]:02d}-{int(day):02d}"  # adjust :02d per MONTHS type

    return None

def parse_downtime(text):
    """Extracts the downtime and returns it in minutes.

    Args:
        text: report text.

    Returns:
        int or None.

    Downtime is written either as 'Stillstand: 150 Minuten' or as
    'Stillstand: 2,5 Stunden', and freetext uses 'stand rund 2,5 Stunden'.
    The unit decides whether the number needs multiplying by 60, and the comma
    has to be replaced before float() will accept it.
    """
    match = re.search(r"(?P<amount>[\d,\.]+)\s*(?P<unit>Minuten|Stunden)", text)
    if not match:
        return None

    amount,unit = match.groups()
    value = float(amount.replace(",","."))

    return int(round(value*60)) if unit == "Stunden" else int(round(value))


def parse_severity(text):
    """Extracts the severity in lower case.

    Args:
        text: report text.

    Returns:
        str or None. One of SEVERITIES.

    The text may write it as 'hoch', 'Hoch' or 'HOCH'. The schema only accepts
    lower case, so the value has to be normalised here.
    """
    match = re.search(
        r"Schweregrad:\s*(?P<value>niedrig|mittel|hoch|kritisch)",
        text, re.IGNORECASE,
    )
    return match.group("value").lower() if match else None


def extract_fields(text):
    """Extracts all fields from one report using regex rules.

    Args:
        text: report text.

    Returns:
        dict with all keys from FIELDS. Values are None where nothing matched.

    Label wording varies: 'Komponente' or 'Bauteil', 'Fehlercode' or
    'Fehler-Nr.', 'Techniker' or 'Bearbeiter'. One pattern per field can cover
    both with an alternation.
    """
    return {
        "report_date": parse_date(text),
        "asset_id": find(r"(?P<value>[A-Z]{2}-\d{4}-[A-Z])",text),
        "component": find(r"(Komponente|Bauteil):\s*(?P<value>.+)",text),
        "error_code": find(r"(Fehlercode|Fehler-Nr\.):\s*(?P<value>E-\d{4})",text),
        "downtime_minutes": parse_downtime(text),
        "severity": parse_severity(text),
        "technician": find(r"(Techniker|Bearbeiter):\s*(?P<value>.+)",text)

    }


def validate_fields(raw):
    """Validates an extracted dict against the ReportFields schema.

    Args:
        raw: dict from extract_fields or from an LLM.

    Returns:
        tuple (fields, errors). Invalid fields are set to None and the valid
        ones kept. errors is a list of the field names that failed.

    A single bad value should not discard the whole report, so only the invalid
    fields are dropped.

    Note the two return paths give different key sets. On success model_dump()
    returns all seven schema fields, filling in None for anything the input did
    not mention. On failure the input dict is copied, so only the keys that
    were passed in come back. extract_fields always supplies all seven, so this
    does not matter here, but an LLM returning a partial object would.
    """
    try:
        return ReportFields(**raw).model_dump(), []
    except ValidationError as e:
        errors = [err["loc"][0] for err in e.errors()]
        fields = {key: (None if key in errors else value) for key, value in raw.items()}
        return fields, errors

def extract_all(reports):
    """Runs extraction over a list of reports.

    Args:
        reports: list of dicts from load_reports.

    Returns:
        list of dicts with keys 'style', 'text', 'truth', 'predicted' and
        'errors'.
    """
    results = []

    for report in reports:
        predictions,errors = validate_fields(extract_fields(report["text"]))
        results.append({
            "style": report["style"],
            "text": report["text"],
            "truth": report["fields"],
            "predicted": predictions,
            "errors": errors
        })
    return results

def score_extraction(results):
    """Scores extraction results per field and per style.

    Args:
        results: list of dicts from extract_all.

    Returns:
        dict with keys 'per_field' (field -> accuracy), 'per_style'
        (style -> accuracy over all fields), 'overall' (float) and
        'n_invalid' (number of reports with at least one validation error).

    A field is correct when the predicted value equals the truth, including
    when both are None. Correctly finding nothing counts as correct.
    """
    field_correct = defaultdict(int)
    field_total = defaultdict(int)
    style_correct = defaultdict(int)
    style_total = defaultdict(int)
    overall_correct = 0
    overall_total   = 0
    n_invalid = 0

    for r in results:
        style = r["style"]
        truth = r["truth"]
        pred = r["predicted"]

        if r["errors"]:
            n_invalid += 1

        for field in FIELDS:
            is_correct = truth[field] == pred.get(field)

            field_total[field] += 1
            style_total[style] += 1
            overall_total += 1

            if is_correct:
                field_correct[field] += 1
                style_correct[style] += 1
                overall_correct += 1

    per_field = {f: field_correct[f] / field_total[f] for f in field_total}
    per_style = {s: style_correct[s] / style_total[s] for s in style_total}
    overall = overall_correct / overall_total if overall_total else 0.0

    return {
        "per_field": per_field,
        "per_style": per_style,
        "overall": overall,
        "n_invalid": n_invalid,
    }

def show_failures(results, field, n=3):
    """Prints reports where one field was extracted wrongly.

    Args:
        results: list of dicts from extract_all.
        field: field name to inspect.
        n: how many to print.
    """
    shown = 0
    for r in results:

        if shown >= n:
            break

        truth = r["truth"].get(field)
        pred = r["predicted"].get(field)

        if truth != pred:
            print(f"[{r['style']}] {field}: truth={truth!r}  predicted={pred!r}")
            print(f"  text: {r['text'][:200]}\n")
            shown += 1


if __name__ == "__main__":
    reports = load_reports()
    print("loaded", len(reports), "reports")

    complete = next(r for r in reports if r["style"] == "complete")
    got = extract_fields(complete["text"])
    assert set(got) == set(FIELDS), set(got) ^ set(FIELDS)
    assert got["asset_id"] == complete["fields"]["asset_id"], got["asset_id"]
    assert got["report_date"] == complete["fields"]["report_date"], got["report_date"]
    assert got["downtime_minutes"] == complete["fields"]["downtime_minutes"]
    print("complete style extracted ok")

    assert parse_date("Bericht vom 2024-10-16") == "2024-10-16"
    assert parse_date("Wartungsbericht vom 19.12.2024") == "2024-12-19"
    assert parse_date("Bericht vom 16. Oktober 2024") == "2024-10-16"
    assert parse_date("kein Datum hier") is None
    print("dates ok")

    assert parse_downtime("Stillstand: 150 Minuten") == 150
    assert parse_downtime("Stillstand: 2,5 Stunden") == 150
    assert parse_downtime("stand rund 4,0 Stunden") == 240
    assert parse_downtime("nichts dazu") is None
    print("downtime ok")

    assert parse_severity("Schweregrad: MITTEL") == "mittel"
    assert parse_severity("Schweregrad: Hoch") == "hoch"
    assert parse_severity("keine Angabe") is None

    fields, errors = validate_fields({"asset_id": "RM-1234-A", "severity": "hoch",
                                      "downtime_minutes": 150})
    assert errors == [], errors
    fields, errors = validate_fields({"asset_id": "XX-1", "severity": "sehr hoch",
                                      "downtime_minutes": 99999})
    assert len(errors) == 3, errors
    assert fields["asset_id"] is None
    print("validation ok")

    results = extract_all(reports)
    assert len(results) == len(reports)

    scores = score_extraction(results)
    assert 0.0 <= scores["overall"] <= 1.0
    assert scores["per_style"]["complete"] > 0.9, scores["per_style"]
    assert scores["per_style"]["complete"] > scores["per_style"]["freetext"]

    print("\nper field:")
    for field, acc in sorted(scores["per_field"].items(), key=lambda kv: kv[1]):
        print(f"  {field:20s} {acc:.3f}")
    print("\nper style:")
    for style, acc in scores["per_style"].items():
        print(f"  {style:10s} {acc:.3f}")
    print(f"\noverall {scores['overall']:.3f}, invalid reports {scores['n_invalid']}")

    worst = min(scores["per_field"], key=scores["per_field"].get)
    print(f"\nworst field: {worst}")
    show_failures(results, worst, n=3)

    print("\nextract.py ran fine.")