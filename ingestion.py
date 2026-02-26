from pathlib import Path
import pypdf
import docx
import pandas as pd



def load_txt(file_path):
    try:
        return file_path.read_text(errors="ignore")
    except:
        return ""


def load_pdf(file_path):
    try:
        reader = pypdf.PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""


def load_docx(file_path):
    try:
        doc = docx.Document(str(file_path))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except:
        return ""

def load_xlsx(file_path):
    try:
        dfs = pd.read_excel(str(file_path), sheet_name=None)

        text = ""

        for sheet_name, df in dfs.items():
            df = df.dropna(how="all")  # remove fully empty rows

            if df.empty:
                continue

            text += f"\nSheet: {sheet_name}\n"
            text += df.fillna("").astype(str).to_string()

        return text.strip()

    except:
        return ""


def build_corpus(base_folder):

    collection = []

    for p in Path(base_folder).rglob("*"):
        if p.suffix.lower() == ".txt":
            collection.append((p.name, load_txt(p)))

        elif p.suffix.lower() == ".pdf":
            collection.append((p.name, load_pdf(p)))

        elif p.suffix.lower() == ".docx":
            collection.append((p.name, load_docx(p)))

        elif p.suffix.lower() in [".xlsx", ".xls"]:
            collection.append((p.name, load_xlsx(p)))

    return collection