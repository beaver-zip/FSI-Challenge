# -*- coding: utf-8 -*-
"""
법령 .doc/.docx 전처리 (본문 + 부칙)
"""
import re
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chardet
from unidecode import unidecode
from tqdm import tqdm
from docx import Document
import textract
import inspect

# =========================
# 경로 / 출력 경로 (수정됨)
# =========================
INPUT_DIR = Path("data/raw/adapter_docs")
OUTPUT_DIR = Path("data/processed")
JSONL_DIR = OUTPUT_DIR / "cleaned_jsonl"
DAPT_TXT = OUTPUT_DIR / "dapt_corpus.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 전처리 옵션 (원본 유지)
# =========================
INCLUDE_ADDENDA_ALL = True
EXCLUDE_OTHER_LAWS_AMENDMENTS = True
EXCLUDE_TOC = False

# =========================
# 패턴/유틸 (원본 유지)
# =========================
CIRCLED_MAP = {
    '①':'1','②':'2','③':'3','④':'4','⑤':'5','⑥':'6','⑦':'7','⑧':'8','⑨':'9','⑩':'10',
    '⑪':'11','⑫':'12','⑬':'13','⑭':'14','⑮':'15','⑯':'16','⑰':'17','⑱':'18','⑲':'19','⑳':'20'
}

HEADER_NOISE_PATTERNS = [
    r"^\s*법제처\s*$",
    r"^\s*국가법령정보센터\s*$",
    r"^페이지\s*\d+/?\d*\s*$",
    r"^\s*\[\s*시행.*\]$",
]

PHONE_PATTERN = r"\b\d{2,4}-\d{3,4}-\d{4}\b|\b1833-\d{4}\b"

ARTICLE_RE = re.compile(r"제\s*([0-9]+(?:[-–][0-9]+)?(?:의[0-9]+)?)\s*조(?:\s*\(([^)]*)\))?")
CHAPTER_RE = re.compile(r"^제\s*([0-9]+)\s*장(?:\s*\(([^)]*)\))?\s*$")
ADDENDA_TITLE_RE = re.compile(r"^\s*부칙\s*(?:\((.*?)\))?\s*$")
ANNEX_RE = re.compile(r"^\s*별표\s*$")
EFFECTIVE_RE = re.compile(r"\\[\s*시행\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*\\]")
LAWNUM_RE = re.compile(r"\\[\s*법률\s*제?(\d+)\s*호.*?\\]")
ADDENDA_BLOCK_RE = re.compile(r"^<\s*제(\d+)호,\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\s*>$ ")

def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', s).strip() or "untitled"

def detect_soffice() -> Optional[str]:
    for cmd in [
        "soffice",
        "/usr/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        r"C:\\Program Files\\LibreOffice\\program\\soffice.exe",
    ]:
        if shutil.which(cmd) or Path(cmd).exists():
            return cmd
    return None

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    return p.returncode, p.stdout, p.stderr

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_doc_with_soffice(path: Path, soffice: str) -> Optional[str]:
    tmp_dir = path.parent / "_tmp_convert"
    tmp_dir.mkdir(exist_ok=True)
    code, _, _ = run([soffice, "--headless", "--convert-to", "docx", "--outdir", str(tmp_dir), str(path)])
    if code == 0:
        converted = tmp_dir / (path.stem + ".docx")
        if converted.exists():
            try:
                return read_docx(converted)
            finally:
                try:
                    converted.unlink()
                except:
                    pass
    return None

def read_doc_with_textract(path: Path) -> Optional[str]:
    try:
        b = textract.process(str(path))
        enc = (chardet.detect(b)["encoding"] or "utf-8")
        return b.decode(enc, errors="ignore")
    except Exception:
        return None

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return read_docx(path)
    if ext == ".doc":
        soffice = detect_soffice()
        if soffice:
            t = read_doc_with_soffice(path, soffice)
            if t:
                return t
        t = read_doc_with_textract(path)
        if t:
            return t
        raise RuntimeError(f".doc 읽기 실패: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")

def normalize_lines(text: str) -> List[str]:
    text = text.replace("\u00A0", " ").replace("\u200b", "")
    text = text.replace("–", "-").replace("―", "-")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    for ln in lines:
        if not ln:
            continue
        if re.search(PHONE_PATTERN, ln):
            continue
        noisy = any(re.match(pat, ln) for pat in HEADER_NOISE_PATTERNS)
        if noisy:
            continue
        out.append(ln)
    return out

def first_nonempty(lines: List[str]) -> str:
    for ln in lines:
        if ln.strip():
            return ln.strip()
    return ""

def parse_basic_meta(lines: List[str], fname: str):
    header_text = "\n".join(lines[:80])
    m = EFFECTIVE_RE.search(header_text)
    eff = date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    m2 = LAWNUM_RE.search(header_text)
    law_no = m2.group(1) if m2 else None

    base = Path(fname).stem
    law_name = re.split(r"[\(\[]", base)[0].strip() or first_nonempty(lines) or base

    if ("시행규칙" in base) or ("시행규칙" in law_name):
        law_type = "RULE"
    elif ("시행령" in base) or ("시행령" in law_name):
        law_type = "DECREE"
    else:
        law_type = "LAW"

    return eff, law_no, law_name, law_type

def status_from_date(eff: Optional[date]) -> str:
    if not eff:
        return "unknown"
    return "current" if eff <= date.today() else "future"

def is_chapter_line(ln: str) -> bool:
    return bool(CHAPTER_RE.match(ln))

def is_article_line(ln: str) -> bool:
    return bool(ARTICLE_RE.search(ln))

def split_main_addenda_lines(lines: List[str]) -> Tuple[List[str], List[str]]:
    main, addenda, in_add = [], [], False
    for ln in lines:
        if ADDENDA_TITLE_RE.match(ln):
            in_add = True
            continue
        (addenda if in_add else main).append(ln)
    return main, addenda

def split_by_articles(lines: List[str]) -> List[Tuple[str, List[str]]]:
    arts = []
    cur_title, cur_buf = None, []
    for ln in lines:
        m = ARTICLE_RE.search(ln)
        if m:
            if cur_title is not None:
                arts.append((cur_title, cur_buf))

            art_no = m.group(1).replace("–", "-")
            art_title = (m.group(2) or "").strip()
            cur_title = f"제{art_no}조" + (f" ({art_title})" if art_title else "")
            tail = ln[m.end():].strip()
            cur_buf = [tail] if tail else []
        else:
            if cur_title is None:
                continue
            cur_buf.append(ln)

    if cur_title is not None:
        arts.append((cur_title, cur_buf))
    return arts

def split_clauses(article_body: List[str]) -> List[Tuple[str, List[str]]]:
    out, cur_key, cur_buf = [], None, []
    for ln in article_body:
        head = ln[:2]
        circled = head and head[0] in CIRCLED_MAP
        num_item = re.match(r"^(\d+)\.\s*", ln)
        kor_item = re.match(r"^([가-힣])\.\s*", ln)

        if circled or num_item:
            if cur_key is not None:
                out.append((cur_key, cur_buf))
            if circled:
                cur_key = head[0]
                ln = ln[1:].strip()
            else: # num_item
                cur_key = num_item.group(1)
                ln = re.sub(r"^\d+\.\s*", "", ln).strip()
            cur_buf = [ln]
        elif kor_item:
            if cur_key is None:
                cur_key, cur_buf = "0", []
            cur_buf.append(ln)
        else:
            if cur_key is None:
                cur_key, cur_buf = "0", []
            cur_buf.append(ln)

    if cur_key is not None:
        out.append((cur_key, cur_buf))
    return out

def parse_addenda_blocks(addenda_lines: List[str]) -> List[Tuple[Optional[str], Optional[str], List[str]]]:
    blocks, cur_meta, cur_buf = [], (None, None), []
    for ln in addenda_lines:
        m = ADDENDA_BLOCK_RE.match(ln)
        if m:
            if cur_buf:
                blocks.append((cur_meta[0], cur_meta[1], cur_buf))
            law_no = m.group(1)
            y, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
            cur_meta = (law_no, f"{y:04d}-{mo:02d}-{d:02d}")
            cur_buf = []
        else:
            cur_buf.append(ln)

    if cur_buf:
        blocks.append((cur_meta[0], cur_meta[1], cur_buf))
    return blocks or [(None, None, addenda_lines)]

def to_structured_records(raw_text: str, src_path: Path) -> List[Dict]:
    lines = normalize_lines(raw_text)
    eff, law_no, law_name, law_type = parse_basic_meta(lines, src_path.name)
    status = status_from_date(eff)

    content_lines, hit_body = [], False
    for ln in lines:
        if is_chapter_line(ln) or is_article_line(ln) or ADDENDA_TITLE_RE.match(ln):
            hit_body = True
        if hit_body:
            content_lines.append(ln)

    main_lines, addenda_lines = split_main_addenda_lines(content_lines)

    if EXCLUDE_TOC:
        main_lines = [ln for ln in main_lines if not re.match(r"^제\d+(?:[-]\d+)?(?:의\d+)?조\s*\S*$", ln)]

    records: List[Dict] = []

    def parse_section(section_name: str, section_lines: List[str], meta_law_no=None, meta_date=None):
        arts = split_by_articles(section_lines)
        for idx, (title, body_lines) in enumerate(arts, start=1):
            m = ARTICLE_RE.search(title)
            art_no = (m.group(1) if m else str(idx)).replace("–", "-")
            art_title = (m.group(2) or "").strip() if m else ""

            if section_name == "ADDENDA" and EXCLUDE_OTHER_LAWS_AMENDMENTS:
                if ("다른 법률의 개정" in art_title) or title.startswith("다른 법률의 개정"):
                    continue

            clauses = split_clauses(body_lines) or [("0", body_lines)]

            for ckey, cbody in clauses:
                clause_no = CIRCLED_MAP.get(ckey, ckey)
                text = re.sub(r"\s+", " ", " ".join(cbody).strip())

                text = re.sub(r"<\s*개정[^>]*>", "", text)
                text = re.sub(r"\\[본조신설.*?\\]", "", text)
                text = re.sub(r"\\[제목개정.*?\\]", "", text)

                rec = {
                    "law_name": law_name,
                    "law_type": law_type,
                    "law_number": law_no,
                    "version": eff.isoformat() if eff else None,
                    "status": status,
                    "section": section_name,
                    "article_no": art_no,
                    "article_title": art_title,
                    "clause_no": clause_no,
                    "text": text,
                    "source_file": str(src_path),
                }
                if section_name == "ADDENDA":
                    rec["addenda_block_law_no"] = meta_law_no
                    rec["addenda_block_date"] = meta_date
                records.append(rec)

    parse_section("MAIN", main_lines)

    if INCLUDE_ADDENDA_ALL:
        blocks = parse_addenda_blocks(addenda_lines)
        for blk_law_no, blk_date, blk_lines in blocks:
            clean_blk = [ln for ln in blk_lines if not re.match(r"^제\d+(?:[-]\d+)?(?:의\d+)?조\s*\S*$", ln)]
            parse_section("ADDENDA", clean_blk, blk_law_no, blk_date)

    return records

def records_to_tagged_text(records: List[Dict]) -> str:
    blocks = []
    for r in records:
        head = (
            f"<law_name>{r['law_name']}</law_name>"
            f"<law_type>{r['law_type']}</law_type>"
            f"<version>{r.get('version') or 'unknown'}</version>"
            f"<status>{r.get('status') or 'unknown'}</status>"
            f"<section>{r.get('section','MAIN')}</section>"
            f"<article_no>{r['article_no']}</article_no>"
            f"<clause_no>{r['clause_no']}</clause_no>"
        )
        body = r["text"]
        if r.get("section") == "ADDENDA":
            meta = ""
            if r.get("addenda_block_law_no") or r.get("addenda_block_date"):
                meta = (f"<addenda_block_law_no>{r.get('addenda_block_law_no','')}</addenda_block_law_no>"
                        f"<addenda_block_date>{r.get('addenda_block_date','')}</addenda_block_date>")
            body = f"{meta}{body}"
        tail = "</text>"
        blocks.append(head + body + tail)
    return "\n\n".join(blocks)

def main():
    DAPT_TXT.write_text("", encoding="utf-8")

    files = sorted([p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in [".doc", ".docx"]])
    if not files:
        print(f"[WARN] {INPUT_DIR}에 .doc/.docx 파일이 없습니다.")
        return

    total_recs = 0
    print(f"[INFO] {len(files)}개 파일 처리 시작 …")
    for fp in tqdm(files):
        try:
            raw = extract_text(fp)
            recs = to_structured_records(raw, fp)
            if not recs:
                print(f"[WARN] 레코드 생성 실패: {fp}")
                continue

            out_jsonl = JSONL_DIR / f"{safe_name(fp.stem)}.jsonl"
            with open(out_jsonl, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            tagged = records_to_tagged_text(recs)
            with open(DAPT_TXT, "a", encoding="utf-8") as f:
                f.write(tagged + "\n\n")

            total_recs += len(recs)

        except Exception as e:
            print(f"[ERROR] {fp.name}: {e}")

    print(f"[DONE] JSONL → {JSONL_DIR}")
    print(f"[DONE] DAPT corpus → {DAPT_TXT}")
    print(f"[INFO] 총 레코드 수: {total_recs}")

if __name__ == "__main__":
    main()