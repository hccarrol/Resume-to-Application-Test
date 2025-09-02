import pathlib, re
import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_lg")

# === (a) Skill Matcher ======================================================
skill_terms = [
    "python", "java", "c++", "machine learning", "deep learning",
    "sql", "excel", "spark", "kubernetes", "react", "pandas",
    "tensorflow", "docker", "aws", "gcp",  # add as needed
]
skill_patterns = [nlp.make_doc(t.lower()) for t in skill_terms]
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_matcher.add("SKILL", None, *skill_patterns)

# === (b) Degree Pattern =====================================================
DEGREES = re.compile(
    r"\b(associate[’']?s?|bachelor[’']?s?|master[’']?s?|mba|ph\.?d\.?|doctorate)\b",
    re.I
)

# === (c) Experience Pattern (years, ranges) =================================
DATE_PAT = re.compile(r"\b(20\d{2}|19\d{2})(\s*[-–]\s*(present|20\d{2}|19\d{2}))?\b", re.I)

# === (d) Job Title Pattern ==================================================
TITLE_PAT = re.compile(r"\b(?:we are looking for|position:|job title:)\s*(.+)", re.I)

def extract_skills(doc):
    matches = skill_matcher(doc)
    return sorted({doc[start:end].text for _, start, end in matches})

def extract_education(text):
    edus = []
    for line in text.splitlines():
        m = DEGREES.search(line)
        if m:
            edus.append(m.group(0))
    return list(set(edus))

def extract_experience(text):
    lines = text.splitlines()
    jobs, buf = [], []
    for ln in lines:
        if DATE_PAT.search(ln) and buf:
            jobs.append(" ".join(buf))
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        jobs.append(" ".join(buf))
    return jobs

def extract_keywords(doc, top_n=10):
    freq = {}
    for tok in doc:
        if tok.is_stop or not tok.is_alpha or tok.pos_ not in {"NOUN", "PROPN", "ADJ"}:
            continue
        lemma = tok.lemma_.lower()
        freq[lemma] = freq.get(lemma, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]

def extract_title(text):
    for line in text.splitlines():
        m = TITLE_PAT.search(line)
        if m:
            return m.group(1).strip()
    return ""

def parse_job_posting(text):
    doc = nlp(text)

    return {
        "job_title": extract_title(text),
        "skills": extract_skills(doc),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "keywords": extract_keywords(doc),
        "raw_text": text
    }

def job_dict_to_row(parsed, file_name=""):
    row = {
        "file_name": file_name,
        "job_title": parsed.get("job_title", ""),
        "skills": ", ".join(parsed.get("skills", [])),
        "keywords": ", ".join(parsed.get("keywords", [])),
        "education": "; ".join(parsed.get("education", [])),
        "experience": "; ".join(parsed.get("experience", [])),
    }
    return row

# === Directory & Output =====================================================
job_dir = pathlib.Path("/Users/holdencarroll/Desktop/PP1/Resume-to-Application-Test/applications")
rows = []

for file in job_dir.glob("*.txt"):
    parsed = parse_job_posting(file.read_text(encoding="utf8"))
    row = job_dict_to_row(parsed, file.name)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("parsed_job_postings.csv", index=False)
print("✓ Job postings parsed:", df.shape, "rows")

#dfCos = pd.read_csv(df)


