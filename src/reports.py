"""Generates synthetic German maintenance reports with known ground truth.

Test data for extract.py. This generator was AI-generated.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_PATH = DATA_DIR / "reports.jsonl"

SEVERITIES = ["niedrig", "mittel", "hoch", "kritisch"]
COMPONENTS = ["Hydraulikpumpe", "Kühlmittelsensor", "Antriebsriemen",
              "Steuerplatine", "Drucklufteinheit", "Getriebelager"]
FINDINGS = ["Druckabfall im Vorlauf festgestellt",
            "Dichtung porös, Austausch erforderlich",
            "Lagergeräusch unter Last hörbar",
            "Verschleiß außerhalb der Toleranz"]
TECHNICIANS = ["M. Kellner", "S. Aydin", "T. Brandt", "J. Novak"]
MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
          "August", "September", "Oktober", "November", "Dezember"]

STYLES = ["complete", "partial", "freetext"]
STYLE_WEIGHTS = [0.55, 0.30, 0.15]


def make_report(rng):
    """Builds one report. Returns dict with 'style', 'text' and 'fields'."""
    d = date(2023, 1, 1) + timedelta(days=rng.randint(0, 900))
    finding = rng.choice(FINDINGS)
    fields = {
        "report_date": d.isoformat(),
        "asset_id": f"RM-{rng.randint(1000, 9999)}-{rng.choice('ABCD')}",
        "component": rng.choice(COMPONENTS),
        "error_code": None if rng.random() < 0.2 else f"E-{rng.randint(1000, 9999)}",
        "severity": rng.choice(SEVERITIES),
        "downtime_minutes": rng.choice([15, 30, 45, 90, 120, 150, 240]),
        "technician": rng.choice(TECHNICIANS),
    }

    style = rng.choices(STYLES, weights=STYLE_WEIGHTS, k=1)[0]
    german_date = f"{d.day:02d}.{d.month:02d}.{d.year}"

    if style == "complete":
        lines = [f"Wartungsbericht vom {german_date}",
                 f"Anlage: {fields['asset_id']}",
                 f"Komponente: {fields['component']}"]
        if fields["error_code"]:
            lines.append(f"Fehlercode: {fields['error_code']}")
        lines += [f"Schweregrad: {fields['severity']}",
                  f"Stillstand: {fields['downtime_minutes']} Minuten",
                  f"Techniker: {fields['technician']}",
                  "", finding + "."]
        text = "\n".join(lines)
        truth = dict(fields)

    elif style == "partial":
        truth = dict(fields)
        lines = []

        if rng.random() < 0.5:
            lines.append(f"Bericht vom {d.day}. {MONTHS[d.month - 1]} {d.year}")
        else:
            lines.append(f"Bericht vom {d.isoformat()}")

        lines.append(f"Anlage: {fields['asset_id']}")

        if rng.random() < 0.4:
            truth["component"] = None
        else:
            lines.append(f"Bauteil: {fields['component']}")

        if fields["error_code"] and rng.random() < 0.7:
            lines.append(f"Fehler-Nr.: {fields['error_code']}")
        else:
            truth["error_code"] = None

        lines.append(f"Schweregrad: {rng.choice([fields['severity'].upper(), fields['severity'].capitalize()])}")

        # downtime sometimes in hours with a comma decimal
        if rng.random() < 0.5:
            hours = fields["downtime_minutes"] / 60
            lines.append(f"Stillstand: {hours:.1f} Stunden".replace(".", ","))
        else:
            lines.append(f"Stillstand: {fields['downtime_minutes']} Minuten")

        if rng.random() < 0.4:
            truth["technician"] = None
        else:
            lines.append(f"Bearbeiter: {fields['technician']}")

        lines += ["", finding + "."]
        text = "\n".join(lines)

    else:  # freetext
        hours = f"{fields['downtime_minutes'] / 60:.1f}".replace(".", ",")
        text = (
            f"Am {d.day}. {MONTHS[d.month - 1]} {d.year} war die Anlage "
            f"{fields['asset_id']} außer Betrieb. {finding}. "
            f"Die Anlage stand rund {hours} Stunden. "
            f"Rückmeldung durch {fields['technician']}."
        )
        truth = dict(fields)
        truth["component"] = None
        truth["error_code"] = None
        truth["severity"] = None

    return {"style": style, "text": text, "fields": truth}


def generate_corpus(n=200, seed=42):
    """Builds n reports from one seeded generator. Returns a list of dicts."""
    rng = random.Random(seed)
    return [make_report(rng) for _ in range(n)]


def save_reports(reports, path=REPORTS_PATH):
    """Writes the reports to a JSONL file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for report in reports:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")


def load_reports(path=REPORTS_PATH):
    """Reads a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    corpus = generate_corpus(n=200, seed=42)
    assert len(corpus) == 200
    save_reports(corpus)
    assert len(load_reports()) == 200

    from collections import Counter
    print(Counter(r["style"] for r in corpus))

    for style in STYLES:
        example = next(r for r in corpus if r["style"] == style)
        print(f"\n--- {style} ---\n{example['text']}")

    print("\nreports.py ran fine.")