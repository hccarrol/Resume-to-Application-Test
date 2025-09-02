import pathlib, re, json
import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher, Matcher

nlp = spacy.load("en_core_web_lg")

# === (a) Skills phrase matcher ===========================================
skill_terms = [
    "python", "java", "c++", "machine learning", "deep learning",
    "sql", "excel", "spark", "kubernetes", "react", "pandas",
    "tensorflow", "docker", "aws", "gcp",  # add as needed
]
skill_patterns = [nlp.make_doc(t.lower()) for t in skill_terms]
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_matcher.add("SKILL", None, *skill_patterns)

# === (b) Education & experience section headers ==========================
SECTION_HEADERS = {
    "education": re.compile(r"^\s*(education|academics|qualifications)\b", re.I),
    "experience": re.compile(r"^\s*(experience|work history|employment)\b", re.I)
}

# === (c) Degree keywords for fine-grained education extraction ===========
DEGREES = re.compile(
    r"\b(associate[’']?s?|bachelor[’']?s?|master[’']?s?|mba|ph\.?d\.?|doctorate)\b",
    re.I
)

def resume_dict_to_row(parsed):
    """
    Turn the dict returned by `parse_resume()` into a flat, CSV-friendly dict.
    """
    row = {}

    # ---- simple scalar fields ---------------------------------------------
    row["file_name"] = parsed.get("file_name", "")

    # ---- list of strings  →  comma-separated ------------------------------
    row["skills"]    = ", ".join(parsed.get("skills", []))
    row["keywords"]  = ", ".join(parsed.get("keywords", []))

    # ---- list of dicts (education)  →  'Degree @ School; …' --------------
    edu_chunks = []
    for edu in parsed.get("education", []):
        degree = edu.get("degree", "").strip()
        inst   = edu.get("institution", "").strip()
        if degree or inst:
            edu_chunks.append(f"{degree} @ {inst}".strip(" @"))
    row["education"] = "; ".join(edu_chunks)

    # ---- list of experience strings  →  semicolon-separated ---------------
    row["experience"] = "; ".join(parsed.get("experience", []))

    return row

def extract_sections(lines):
    """Return dict with 'education', 'experience', 'other' lists of lines."""
    current = "other"
    buckets = {"education": [], "experience": [], "other": []}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        sw = False
        for sec, pat in SECTION_HEADERS.items():
            if pat.match(stripped):
                current = sec
                sw = True
                break
        if not sw:
            buckets[current].append(stripped)
    return buckets


def find_skills(doc):
    """Return unique skill keywords matched in doc."""
    matches = skill_matcher(doc)
    return sorted({doc[start:end].text for _, start, end in matches})


def parse_education(lines):
    """Extract degree + institution heuristically."""
    edus = []
    for ln in lines:
        m = DEGREES.search(ln)
        if m:
            degree = m.group(0)
            # grab the rest of the line as institution/program guess
            rest = ln[m.end():].strip(" ,;-")
            edus.append({"degree": degree, "institution": rest or None, "raw": ln})
    return edus


def parse_experience(lines):
    """Very naive job-block detector: Year–Year or Year–Present lines."""
    jobs, buf = [], []
    date_pat = re.compile(r"\b(20\d{2}|19\d{2})\b")
    for ln in lines:
        if date_pat.search(ln) and buf:
            jobs.append(" ".join(buf))
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        jobs.append(" ".join(buf))        # add final block
    return jobs


def parse_resume(text):
    lines = text.splitlines()
    sections = extract_sections(lines)

    # spaCy doc for entire resume (for skills & keywords)
    doc = nlp(text)

    data = {
        "skills": find_skills(doc),                        # list[str]
        "education": parse_education(sections["education"]),  # list[dict]
        "experience": parse_experience(sections["experience"]),# list[str]
        "raw_text": text,
    }

    # optional: extract top-n nouns/adjectives as "keywords"
    top_n = 10
    tok_freq = {}
    for tok in doc:
        if tok.is_stop or not tok.is_alpha or tok.pos_ not in {"NOUN", "PROPN", "ADJ"}:
            continue
        tok_freq[tok.lemma_.lower()] = tok_freq.get(tok.lemma_.lower(), 0) + 1
    data["keywords"] = sorted(tok_freq, key=tok_freq.get, reverse=True)[:top_n]

    return data


resume_dir = pathlib.Path("/Users/holdencarroll/Desktop/PP1/Resume-to-Application-Test/resumes")
rows = []

for file in resume_dir.glob("*.txt"):
    parsed = parse_resume(file.read_text(encoding="utf8"))
    parsed["file_name"] = file.name          # keep the source
    rows.append(resume_dict_to_row(parsed))

df = pd.DataFrame(rows)
df.to_csv("parsed_resumes.csv", index=False)
print("✓ CSV written:", df.shape, "rows")