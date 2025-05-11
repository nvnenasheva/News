import pandas as pd
import numpy as np
import re
import requests
import csv
from bs4 import BeautifulSoup

from collections import Counter
from rouge_score import rouge_scorer

from tqdm import tqdm 
import re, difflib, unicodedata
from urllib.parse import urlparse
from typing import List, Tuple

import matplotlib.pyplot as plt

_marker = re.compile(
    r'''
        \s*(?:                # пробелы слева/справа и любой из вариантов…
            ={2,}             # ===, ====, =====
          | -{2,}             # ---, ----, …
          | \bВРЕЗКА\b
          | \bКОНЕЦ\s+ВРЕЗКИ\b
          | \bA:\b
          | \bQ:\b
        )\s*
    ''', re.IGNORECASE | re.VERBOSE)

__DASHES = {0x2014: "-", 0x2013: "-", 0x2212: "-", 0x2011: "-", 0x2010: "-"}
_TRANS  = str.maketrans({chr(k): v for k, v in __DASHES.items()})

# ─────────────────────────────────────────────────────────
# Загрузка по URL
# ─────────────────────────────────────────────────────────

def process_row(line):
    #print(f"Скачиваем текст по ссылке: {line}")
    reference_text = get_text_from_url(line["URL"])
    if len(reference_text)==0:
        print("[ERROR]: Проблема с извлечением текста по ссылке: ", "<<",line["URL"],">>, \n")
    
    return pd.Series({"reference_text": reference_text})

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = _marker.sub(' ', text)          # убираем все маркеры
    text = re.sub(r'\s+', ' ', text)       # схлопываем пробелы и переводы строк
    return text.strip()

def get_text_from_url(url):
    try:
        # скачиваем страницу
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding 

        # парсим страницу
        soup = BeautifulSoup(response.text, 'html.parser')

        # пробуем сначала <article>, затем fallback на <p>
        article = soup.find('article')
        if article:
            full_text = article.get_text(separator=' ')
            #print(full_text)
        else:
            paragraphs = soup.find_all('p')
            full_text = ' '.join(p.get_text() for p in paragraphs)
            
        return clean_text(full_text)
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return ""
        # Readability: часто сразу даёт чистый текст
        doc       = Document(html)
        main_html = doc.summary(html_partial=True)
        text      = clean_text(BeautifulSoup(main_html, "html.parser")
                               .get_text(" "))

        # fallback: если извлеклось слишком мало символов
        if len(text) < 200:                           # эвристика
            soup  = BeautifulSoup(html, "html.parser")
            text  = clean_text(" ".join(
                p.get_text(" ", strip=True) for p in soup.find_all("p")
            ))

        return text

    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return ""

# ─────────────────────────────────────────────────────────
# Тип источника новости
# ─────────────────────────────────────────────────────────

def get_domain(url: str) -> str:
    netloc = urlparse(url.strip()).netloc.lower()      # m.lenta.ru → m.lenta.ru
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # уберём субдомен «m.» или «amp.»
    parts = netloc.split(".")
    if len(parts) > 2:                                 # m.lenta.ru → lenta.ru
        netloc = ".".join(parts[-2:])
    return netloc


# ─────────────────────────────────────────────────────────
# Поиск подстрок
# ─────────────────────────────────────────────────────────


def _normalize(txt: str) -> str:
    #юникод‑NFC + одинаковые тире.
    txt = unicodedata.normalize("NFC", txt).translate(_TRANS)
    return txt.replace("\u00A0", " ").replace("\u200B", "")

# разбиваем текст с маркерами → [(sentence, label), …]
LINE_MARK = re.compile(r"^\s*(={3,}|-{3,})\s*(.*?)\s*(?:=+|-+)?\s*$")

def parse_markup(text: str) -> List[Tuple[str, str]]:
    """
    • `=== Заголовок ===`       → label='title'
    • `--- Подзаголовок ---`    → label='subtitle'
    • блок между
         === ВРЕЗКА … ===
         === КОНЕЦ ВРЕЗКИ ===   → label='vrezka'
    • остальное                 → label='body'
    """
    lines      = _normalize(text).splitlines()
    in_vrezka  = False
    chunks: List[Tuple[str, str]] = []

    def _add(sent: str, lbl: str):
        for s in re.split(r'(?<=[\.\?!…])\s+', sent):
            s = s.strip()
            if s:
                chunks.append((s, lbl))

    for ln in lines:
        m = LINE_MARK.match(ln)
        if m:
            tag = m.group(2).strip()      # сам заголовок без === / ---
            # начало или конец врезки?
            if tag.upper().startswith("ВРЕЗКА"):
                in_vrezka = True
                continue                   # метку самой строки не сохраняем
            if tag.upper().startswith("КОНЕЦ ВРЕЗКИ"):
                in_vrezka = False
                continue
            # обычные заголовки
            lbl = 'title' if ln.lstrip().startswith('===') else 'subtitle'
            _add(tag, lbl)
            continue

        # обычный текст
        lbl = 'vrezka' if in_vrezka else 'body'
        _add(ln, lbl)

    return chunks


def clean_ws(text: str, *, keep_newlines: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TRANS)
    text = text.replace("\u00A0", " ").replace("\u200B", "")

    if keep_newlines:
        horiz = r"[ \t\f\v]+"                    # любые «горизонтальные» пробелы
        text = re.sub(horiz, " ", text)          # мн. пробелы → 1
        text = re.sub(r"\n" + horiz, "\n", text) # пробелы *после* \n
        text = re.sub(horiz + r"\n", "\n", text) # пробелы *перед* \n  ← NEW
        text = re.sub(r"\n{2,}", "\n", text)     # мн. \n → 1
    else:
        text = re.sub(r"\s+", " ", text)         # всё whitespace → пробел

    return text.strip()

def sentence_split(text: str):
    text = clean_ws(text, keep_newlines=True)

    primary = re.split(r'(?<=[\.\?!…])\s+', text)       # конец предложения
    sentences = []
    for chunk in primary:
        sentences.extend(s.strip() for s in chunk.split('\n') if s.strip())
    return sentences


def find_missing_extra(
        ref_chunks,              # [(sent, lbl), …]  – parse_markup(ref)
        ext_chunks,              # [(sent, lbl), …]  – parse_markup(ext)
        min_words: int = 5):

    # раздельные списки предложений и меток
    ref_sents, ref_lbls = zip(*ref_chunks)
    ext_sents, ext_lbls = zip(*ext_chunks)

    sm = difflib.SequenceMatcher(None, ref_sents, ext_sents, autojunk=False)
    missing, extra = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('delete', 'replace'):          # пропавшие
            for k in range(i1, i2):
                s = ref_sents[k]
                if len(s.split()) >= min_words and s not in ext_sents[j1:j2]:
                    missing.append((s, ref_lbls[k]))

        if tag in ('insert', 'replace'):          # лишние
            for k in range(j1, j2):
                s = ext_sents[k]
                if len(s.split()) >= min_words and s not in ref_sents[i1:i2]:
                    extra.append((s, ext_lbls[k]))

    return missing, extra


def diff_row(row, *,   # ← вызывается из .apply
             min_words: int = 5,
             keep_labels: bool = False      # нужно ли хранить метки в колонке
            ) -> pd.Series:

    # row['reference']  – эталон с маркерами (=== … ===, --- … --- и т.д.)
    # row['extracted']  – извлечённый текст

    ref_chunks = parse_markup(row['reference_text'])
    ext_chunks = parse_markup(row['extracted_text'])

    missing, extra = find_missing_extra(ref_chunks, ext_chunks,
                                        min_words=min_words)

    # --- считаем, сколько «важного» пропало (пример) --------------
    lost_title   = sum(lbl == 'title'   for _, lbl in missing)
    lost_vrezka  = sum(lbl == 'vrezka'  for _, lbl in missing)
    lost_body  = sum(lbl == 'body'  for _, lbl in missing)

    # --- отдаём Series с нужными полями ---------------------------
    out = {
        'missing_chunks': [m if keep_labels else m[0] for m in missing],
        'extra_chunks'  : extra,
        'missing_cnt'   : len(missing),
        'extra_cnt'     : len(extra),
        'lost_title_cnt': lost_title,
        'lost_vrezka_cnt': lost_vrezka,
        'lost_body_cnt': lost_body,
    }
    return pd.Series(out)

# ─────────────────────────────────────────────────────────
# Метрики: ьаза
# ─────────────────────────────────────────────────────────

PAT = re.compile(
    r'''
        (?:={2,}|-{2,})         # ===   или ----
      | \bВРЕЗКА\b
      | \bКОНЕЦ\s+ВРЕЗКИ\b
      | \bA:\b
      | \bQ:\b
    ''',
    flags=re.IGNORECASE | re.VERBOSE
)

def tokenize(text: str):
    # делим строку на слова, убираем заглавные буквы
    return re.findall(r'\b\w+\b', text.lower())    


def clean_series(col: pd.Series, *, keep_newlines: bool = True) -> pd.Series:
    horiz_ws = r"[ \t\f\v]+"       # «горизонтальный» пробел без \n

    cleaned = (
        col.fillna("")                                            # NaN → ''
           .str.normalize("NFC")                                  # канонизация Юникода
           .str.translate(_TRANS)                                # тире → дефис
           .str.replace("\u00A0", " ", regex=False)              # неразрывный пробел
           .str.replace("\u200B", "",  regex=False)              # zero‑width space
           .str.replace(PAT, " ", regex=True)                    # вырезаем «маркеры»
    )

    if keep_newlines:
        cleaned = (
            cleaned
              .str.replace(horiz_ws, " ", regex=True)             # мн. пробелы → 1
              .str.replace(r"\n" + horiz_ws, "\n", regex=True)    # пробелы после \n
              .str.replace(r"\n{2,}", "\n", regex=True)           # мн. \n → 1
        )
    else:
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)    # всё whitespace → пробел

    cleaned = cleaned.str.strip()

    return cleaned.replace("", np.nan)                            # '' → NaN
    

def prf(reference_text: str, extracted_text: str):
    # precision, recall, f1 на основе перекрытия слов
    ref_tokens  = tokenize(reference_text)
    extr_tokens = tokenize(extracted_text)

    # как часто появляются слова в тексте
    ref_counts  = Counter(ref_tokens)
    extr_counts = Counter(extr_tokens)

    tp=0
    for word, ref_freq in ref_counts.items():
        tp += min(ref_freq, extr_counts.get(word, 0)) # пусть слово 'co' встретилось в REF 5 раза, а в EXTR -- 10 => за true считаем меньшее из этих чисел

    ref_total  = sum(ref_counts.values())  # всего слов в REF
    extr_total = sum(extr_counts.values())  # всего слов в EXTR

    recall    = tp / ref_total  if ref_total  else 0.0
    precision = tp / extr_total if extr_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1

def jaccard_similarity(reference: str, extracted: str) -> float:
    # J(A, B) = |A ∩ B| / |A ∪ B|, где A и B — множества уникальных слов двух текстов.
    set_ref = set(tokenize(reference))
    set_extr = set(tokenize(extracted))

    if not set_ref and not set_extr:     # оба текста пусты
        return 0.0

    intersection = set_ref & set_extr
    union        = set_ref | set_extr
    
    return len(intersection) / len(union)


def basic_metrics(line):
    reference_text = line['reference_text']
    extracted_text = line['extracted_text']
    
    p, r, f1 = prf(reference_text, extracted_text)
    j = jaccard_similarity(reference_text, extracted_text)
    
    return pd.Series({"extr_len": len(line["extracted_text"]),
                      "ref_len": len(reference_text),
                      "p": p, "r": r, "f1": f1,
                      "jaccard": j})
# ─────────────────────────────────────────────────────────
# Метрики: не база (смысловое)
# ─────────────────────────────────────────────────────────

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def rouge_row(row):
    ref_chunks = parse_markup(row['reference_text'])
    ext_chunks = parse_markup(row['extracted_text'])

    ref_plain = " ".join(s for s, _ in ref_chunks)
    ext_plain = " ".join(s for s, _ in ext_chunks)

    score = scorer.score(ref_plain, ext_plain)['rougeL']
    return pd.Series({
        'rougeL_r': score.recall,
        'rougeL_p': score.precision,
        'rougeL_f': score.fmeasure,
    })