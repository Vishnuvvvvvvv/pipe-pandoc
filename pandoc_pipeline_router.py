"""
pandoc_pipeline_router.py thats it
─────────────────────────
Self-contained FastAPI router — drop this single file into any FastAPI project.
Uses PyPandoc (Pandoc) as the primary engine for BOTH Markdown extraction and DOM traversal.
This ensures perfect structural fidelity for nested tables, colspans/rowspans, and hierarchical lists.

USAGE in your existing main.py:
    from pandoc_pipeline_router import router as pandoc_router
    app.include_router(pandoc_router, prefix="/pandoc", tags=["Pandoc Pipeline"])
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
import time as _time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
import pypandoc

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("pandoc_pipeline")

# ── Config from environment ────────────────────────────────────────────────────
AWS_REGION             = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID      = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY", "")
BEDROCK_MODEL_ID       = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
USE_VLM                = os.getenv("USE_VLM", "true").lower() == "true"
USE_TEXTRACT           = os.getenv("USE_TEXTRACT", "true").lower() == "true"
OUTPUT_DIR             = Path(os.getenv("DOCX_OUTPUT_DIR", "./uploads"))
IMAGES_SCALE           = 2.0
TEXTRACT_MAX_BYTES     = 5 * 1024 * 1024
MIN_CHART_AREA         = 40_000
MIN_OCR_TOKENS         = 2
MAX_OCR_TOKENS         = 80
ALLOWED_EXT            = {".doc", ".docx"}
STAGING_DIR            = OUTPUT_DIR / "_staging"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

# ── Pydantic schemas ───────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    document: Optional[str] = None
    tables_count: Optional[int] = None
    images_count: Optional[int] = None
    markdown_preview: Optional[str] = None
    markdown_path: Optional[str] = None
    semantic_json_path: Optional[str] = None
    dom_json_path: Optional[str] = None
    error: Optional[str] = None

# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter()

def _aws_kwargs() -> dict:
    kw: dict[str, Any] = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kw["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kw["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return kw

# ══════════════════════════════════════════════════════════════════════════════
# OCR — Amazon Textract
# ══════════════════════════════════════════════════════════════════════════════
def _img_to_bytes_for_textract(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    if len(data) <= TEXTRACT_MAX_BYTES:
        return data
    for q in (90, 75, 60, 45):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        data = buf.getvalue()
        if len(data) <= TEXTRACT_MAX_BYTES:
            return data
    return data

def run_ocr(img: Image.Image) -> str:
    if not USE_TEXTRACT:
        return ""
    try:
        image_bytes = _img_to_bytes_for_textract(img)
        client = boto3.client("textract", **_aws_kwargs())
        response = client.detect_document_text(Document={"Bytes": image_bytes})
        lines = [b["Text"] for b in response.get("Blocks", []) if b.get("BlockType") == "LINE"]
        return "\n".join(lines).strip()
    except Exception as exc:
        log.warning("OCR error: %s", exc)
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# CHART DETECTION — AWS Bedrock Nova Lite
# ══════════════════════════════════════════════════════════════════════════════
_CHART_PROMPT = (
    "You are analyzing an image extracted from a business document. "
    "Determine whether the image is a chart, graph, diagram, or table of data. "
    "Return ONLY valid JSON — no prose, no markdown fences — matching this shape:\n"
    '{"chart_type": "<bar|line|pie|table|diagram|not_a_chart>", '
    '"series": [{"label": "<string>", "value": "<string>"}], '
    '"summary": "<one sentence describing the chart>"}\n'
    "If the image is a photo, logo, signature, or decorative element, "
    'return {"chart_type": "not_a_chart", "series": [], "summary": ""}.'
)
_NOT_EVALUATED = {"chart_type": "not_evaluated", "series": [], "summary": ""}
_ERROR_RESULT   = {"chart_type": "error",         "series": [], "summary": ""}

def looks_like_chart(img: Image.Image, ocr_text: str) -> bool:
    area = img.width * img.height
    tokens = len(ocr_text.split())
    return area > MIN_CHART_AREA and MIN_OCR_TOKENS <= tokens <= MAX_OCR_TOKENS

def describe_chart(image_path: Path) -> dict[str, Any]:
    try:
        with Image.open(image_path) as im:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        client = boto3.client("bedrock-runtime", **_aws_kwargs())
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [
                {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                {"text": _CHART_PROMPT}
            ]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return parsed
    except Exception as exc:
        log.warning("VLM error: %s", exc)
        return {**_ERROR_RESULT, "summary": str(exc)}

# ══════════════════════════════════════════════════════════════════════════════
# PYPANDOC AST TRAVERSAL & EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _find_tables_in_ast(ast_node: Any, tables_list: list) -> None:
    if isinstance(ast_node, dict):
        if ast_node.get("t") == "Table":
            tables_list.append(ast_node)
        for val in ast_node.values():
            _find_tables_in_ast(val, tables_list)
    elif isinstance(ast_node, list):
        for item in ast_node:
            _find_tables_in_ast(item, tables_list)

def extract_tables_from_ast(ast: dict) -> list[dict[str, Any]]:
    raw_tables = []
    _find_tables_in_ast(ast, raw_tables)
    return [{"table_index": i, "raw_pandoc_ast": t_node} for i, t_node in enumerate(raw_tables)]

def process_extracted_images(image_dir: Path) -> list[dict[str, Any]]:
    if not image_dir.exists():
        return []
    out = []
    for idx, img_path in enumerate(sorted(image_dir.rglob("*"))):
        if not img_path.is_file(): continue
        try:
            with Image.open(img_path) as img:
                img_copy = img.copy()
            ocr_text = run_ocr(img_copy)
            semantic = describe_chart(img_path) if (USE_VLM and looks_like_chart(img_copy, ocr_text)) else _NOT_EVALUATED.copy()
            out.append({
                "picture_index": idx,
                "file": str(img_path),
                "width": img_copy.width,
                "height": img_copy.height,
                "ocr_text": ocr_text,
                "semantic": semantic,
            })
        except Exception as exc:
            log.warning("Could not process image %s: %s", img_path.name, exc)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_markdown(base_md: str, tables: list, images: list) -> str:
    parts = [base_md]
    if images:
        parts.append("\n\n---\n## Extracted Image & Chart Data\n")
        for rec in images:
            fname = Path(rec["file"]).name
            parts.append(f"\n### Image {rec['picture_index']}  (`{fname}`)")
            if rec.get("ocr_text"):
                parts.append(f"\n**OCR text:**\n```\n{rec['ocr_text']}\n```")
            sem = rec.get("semantic", {})
            ct = sem.get("chart_type", "")
            if ct and ct not in ("not_evaluated", "not_a_chart", "unavailable", "error"):
                parts.append(f"\n**Chart type:** `{ct}`")
                if sem.get("summary"): parts.append(f"**Summary:** {sem['summary']}")
                for pt in sem.get("series", []):
                    parts.append(f"- {pt.get('label','')}: {pt.get('value','')}")
    return "\n".join(parts)

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try: return super().default(obj)
        except TypeError: return str(obj)

# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Mapping: Word style name → (numId for that list family, ilvl for nesting depth)
# These are STANDARD Word built-in style names. Documents using custom styles
# will fall through gracefully and be processed as-is.
_WORD_LIST_STYLE_MAP = {
    # Bullet families (all share same list, just deeper ilvl)
    "ListBullet":   ("bullet_family", "0"),
    "ListBullet2":  ("bullet_family", "1"),
    "ListBullet3":  ("bullet_family", "2"),
    "ListBullet4":  ("bullet_family", "3"),
    "ListBullet5":  ("bullet_family", "4"),
    # Number families
    "ListNumber":   ("number_family", "0"),
    "ListNumber2":  ("number_family", "1"),
    "ListNumber3":  ("number_family", "2"),
    "ListNumber4":  ("number_family", "3"),
    "ListNumber5":  ("number_family", "4"),
}


def _preprocess_list_nesting(docx_path: Path) -> Path:
    """
    Pre-processor that fixes nested lists encoded via Word's built-in
    ListBullet2 / ListBullet3 / ListNumber2 style names.

    Pandoc does not recognize these style suffixes as nesting depth cues,
    so it outputs everything as a flat Level-0 list.

    This function:
    1. Reads styles.xml to discover the real numId for ListBullet and ListNumber.
    2. Ensures numbering.xml has levels 0-4 defined for those numIds.
    3. Rewrites each paragraph's w:numPr so the ilvl matches the style suffix.
    4. Returns path to the fixed DOCX (original is unchanged).
    """
    import xml.dom.minidom as minidom

    try:
        with zipfile.ZipFile(docx_path, "r") as zin:
            names = set(zin.namelist())
            if "word/document.xml" not in names:
                return docx_path

            # ── 1. Read styles.xml to get numId for bullet + number families ──
            bullet_numid = "1"
            number_numid = "5"
            if "word/styles.xml" in names:
                sdom = minidom.parseString(zin.read("word/styles.xml"))
                for style in sdom.getElementsByTagName("w:style"):
                    sid = style.getAttribute("w:styleId")
                    numIds = style.getElementsByTagName("w:numId")
                    if not numIds:
                        continue
                    nid = numIds[0].getAttribute("w:val")
                    if sid == "ListBullet" and nid:
                        bullet_numid = nid
                    elif sid == "ListNumber" and nid:
                        number_numid = nid

            # Build resolved map: style_name → (actual_numId, ilvl)
            resolved_map: dict[str, tuple[str, str]] = {}
            for style_name, (family, ilvl) in _WORD_LIST_STYLE_MAP.items():
                nid = bullet_numid if family == "bullet_family" else number_numid
                resolved_map[style_name] = (nid, ilvl)

            # ── 2. Patch numbering.xml to ensure levels 1-4 exist ──
            num_data_original = None
            num_data_patched = None
            if "word/numbering.xml" in names:
                num_data_original = zin.read("word/numbering.xml")
                ndom = minidom.parseString(num_data_original)

                # numId → abstractNumId
                num_to_abstract: dict[str, str] = {}
                for num in ndom.getElementsByTagName("w:num"):
                    nid = num.getAttribute("w:numId")
                    refs = num.getElementsByTagName("w:abstractNumId")
                    if refs:
                        num_to_abstract[nid] = refs[0].getAttribute("w:val")

                # For each relevant numId, ensure levels 1-4 exist in its abstractNum
                for nid in {bullet_numid, number_numid}:
                    abstract_id = num_to_abstract.get(nid)
                    if not abstract_id:
                        continue
                    for an in ndom.getElementsByTagName("w:abstractNum"):
                        if an.getAttribute("w:abstractNumId") != abstract_id:
                            continue
                        existing_levels = {
                            lvl.getAttribute("w:ilvl")
                            for lvl in an.getElementsByTagName("w:lvl")
                        }
                        # Detect format from level 0
                        fmt = "bullet"
                        for lvl in an.getElementsByTagName("w:lvl"):
                            if lvl.getAttribute("w:ilvl") == "0":
                                nfmts = lvl.getElementsByTagName("w:numFmt")
                                if nfmts:
                                    fmt = nfmts[0].getAttribute("w:val")
                                break
                        for lvl_num in range(1, 5):
                            if str(lvl_num) in existing_levels:
                                continue
                            lvl_el = ndom.createElement("w:lvl")
                            lvl_el.setAttribute("w:ilvl", str(lvl_num))
                            start_el = ndom.createElement("w:start")
                            start_el.setAttribute("w:val", "1")
                            nfmt_el = ndom.createElement("w:numFmt")
                            nfmt_el.setAttribute("w:val", fmt)
                            ltxt_el = ndom.createElement("w:lvlText")
                            ltxt_el.setAttribute(
                                "w:val", "%1." if fmt == "decimal" else "\u2022"
                            )
                            ljc_el = ndom.createElement("w:lvlJc")
                            ljc_el.setAttribute("w:val", "left")
                            pPr_el = ndom.createElement("w:pPr")
                            ind_el = ndom.createElement("w:ind")
                            ind_el.setAttribute(
                                "w:left", str(720 + lvl_num * 720)
                            )
                            ind_el.setAttribute("w:hanging", "360")
                            pPr_el.appendChild(ind_el)
                            for child in [start_el, nfmt_el, ltxt_el, ljc_el, pPr_el]:
                                lvl_el.appendChild(child)
                            an.appendChild(lvl_el)
                num_data_patched = ndom.toxml().encode("utf-8")

            # ── 3. Patch document.xml to inject correct w:numPr ──
            doc_data = zin.read("word/document.xml")
            ddom = minidom.parseString(doc_data)
            modified = False
            for p in ddom.getElementsByTagName("w:p"):
                pPrs = p.getElementsByTagName("w:pPr")
                if not pPrs:
                    continue
                pPr = pPrs[0]
                styles = pPr.getElementsByTagName("w:pStyle")
                if not styles:
                    continue
                style_val = styles[0].getAttribute("w:val")
                if style_val not in resolved_map:
                    continue
                numId_val, ilvl_val = resolved_map[style_val]
                # Remove any existing numPr so we can inject a clean one
                for old in list(pPr.getElementsByTagName("w:numPr")):
                    pPr.removeChild(old)
                numPr_el = ddom.createElement("w:numPr")
                ilvl_el = ddom.createElement("w:ilvl")
                ilvl_el.setAttribute("w:val", ilvl_val)
                numId_el = ddom.createElement("w:numId")
                numId_el.setAttribute("w:val", numId_val)
                numPr_el.appendChild(ilvl_el)
                numPr_el.appendChild(numId_el)
                pPr.appendChild(numPr_el)
                modified = True

            if not modified:
                # No changes needed — use original file
                return docx_path

            # ── 4. Write patched DOCX to a temp file ──
            fixed_path = docx_path.parent / f"_fixed_{docx_path.name}"
            with zipfile.ZipFile(docx_path, "r") as zin2:
                with zipfile.ZipFile(fixed_path, "w", zipfile.ZIP_DEFLATED) as zout:
                    for item in zin2.infolist():
                        if item.filename == "word/document.xml":
                            zout.writestr(item, ddom.toxml().encode("utf-8"))
                        elif item.filename == "word/numbering.xml" and num_data_patched:
                            zout.writestr(item, num_data_patched)
                        else:
                            zout.writestr(item, zin2.read(item.filename))

            return fixed_path

    except Exception as exc:
        log.warning("List-nesting pre-processor failed (%s) — using original file", exc)
        return docx_path


def _process_document(docx_path: Path) -> dict[str, Any]:
    name = docx_path.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "images"
    media_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing with Pandoc: %s", docx_path.name)

    # Pre-process: fix ListBullet2/ListNumber2 style nesting before Pandoc sees it
    fixed_path = _preprocess_list_nesting(docx_path)
    use_path = fixed_path  # may be same as docx_path if nothing was modified

    # 1. Generate Markdown using Pandoc
    try:
        base_md = pypandoc.convert_file(
            str(use_path),
            'gfm',
            extra_args=['--wrap=none']
        )
        import re
        # Strip empty HTML comments Pandoc injects between loose lists
        base_md = re.sub(r'(\r?\n)*<!-- -->(\r?\n)*', r'\n', base_md)
        # Compress double newlines inside HTML blocks so tables render
        base_md = re.sub(r'>\n{2,}<', '>\n<', base_md)
        # Unescape dollar signs (breaks table rendering in some viewers)
        base_md = base_md.replace(r'\$', '$')
    except Exception as exc:
        log.error("Pandoc markdown extraction failed: %s", exc)
        return {"success": False, "document": docx_path.name, "error": f"Pandoc error: {exc}"}
    finally:
        # Clean up the temp fixed file if we created one
        if fixed_path != docx_path:
            try:
                fixed_path.unlink(missing_ok=True)
            except Exception:
                pass

    # 2. Extract DOM (AST) — use original path for JSON AST
    dom_ast = {}
    try:
        dom_json_str = pypandoc.convert_file(str(docx_path), 'json')
        dom_ast = json.loads(dom_json_str)
    except Exception as exc:
        log.warning("PyPandoc DOM extraction failed: %s", exc)

    # Extract media using Zipfile
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx_zip:
            media_files = [n for n in docx_zip.namelist() if n.startswith("word/media/")]
            for m in media_files:
                if not m.endswith('/'):
                    dest = media_dir / Path(m).name
                    dest.write_bytes(docx_zip.read(m))
    except Exception as exc:
        log.warning("Zipfile media extraction failed: %s", exc)

    dom_path = out_dir / f"{name}.dom.json"
    try:
        dom_path.write_text(json.dumps(dom_ast, indent=2, cls=_SafeEncoder), encoding="utf-8")
    except Exception:
        pass

    # 3. Process Tables + Images
    tables = extract_tables_from_ast(dom_ast) if dom_ast else []
    images = process_extracted_images(media_dir)

    semantic_path = out_dir / f"{name}.semantic.json"
    semantic = {
        "document": docx_path.name,
        "schema_version": "1.0",
        "tables": tables,
        "images": images,
    }
    semantic_path.write_text(json.dumps(semantic, indent=2, cls=_SafeEncoder), encoding="utf-8")

    full_md = render_markdown(base_md, tables, images)
    md_path = out_dir / f"{name}.md"
    md_path.write_text(full_md, encoding="utf-8")

    log.info("Done: %s  (tables=%d, images=%d)", docx_path.name, len(tables), len(images))

    return {
        "success": True,
        "document": docx_path.name,
        "error": None,
        "markdown": full_md,
        "dom_json_path": str(dom_path),
        "semantic_json_path": str(semantic_path),
        "markdown_path": str(md_path),
        "tables_count": len(tables),
        "images_count": len(images),
    }

def _run_pipeline(job_id: str, docx_path: Path) -> None:
    _jobs[job_id]["status"] = "processing"
    try:
        result = _process_document(docx_path)
        _jobs[job_id].update({
            "status": "done" if result["success"] else "error",
            "result": result,
        })
    except Exception as exc:
        log.exception("Pipeline crashed for job %s", job_id)
        _jobs[job_id].update({
            "status": "error",
            "result": {"success": False, "error": str(exc)},
        })
    finally:
        try: docx_path.unlink(missing_ok=True)
        except Exception: pass

def _convert_doc_to_docx(doc_path: Path) -> Optional[Path]:
    out_path = STAGING_DIR / (doc_path.stem + ".docx")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from doc2docx import convert
        convert(str(doc_path), str(out_path))
        if out_path.exists(): return out_path
    except Exception:
        pass
    import subprocess
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        try:
            subprocess.run(["soffice", "--headless", "--convert-to", "docx", "--outdir", str(STAGING_DIR), str(doc_path)], check=True)
            if out_path.exists(): return out_path
        except Exception: pass
    return None

@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Only .doc and .docx accepted. Got: '{suffix}'")
    job_id = str(uuid.uuid4())
    tmp_path = STAGING_DIR / f"{job_id}{suffix}"
    try:
        with tmp_path.open("wb") as fh: shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()
    if suffix == ".doc":
        tmp_path = _convert_doc_to_docx(tmp_path)
        if tmp_path is None: raise HTTPException(status_code=422, detail="Could not convert .doc")
    _jobs[job_id] = {"status": "queued", "result": None, "filename": file.filename}
    background_tasks.add_task(_executor.submit, _run_pipeline, job_id, tmp_path)
    return UploadResponse(job_id=job_id, status="queued", message=f"'{file.filename}' queued. Poll /status/{job_id}")

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None: raise HTTPException(status_code=404, detail="Job not found.")
    result = job.get("result") or {}
    md = result.get("markdown", "")
    return JobStatus(
        job_id=job_id, status=job["status"], document=result.get("document") or job.get("filename"),
        tables_count=result.get("tables_count"), images_count=result.get("images_count"),
        markdown_preview=md[:2000] if md else None, markdown_path=result.get("markdown_path"),
        semantic_json_path=result.get("semantic_json_path"), dom_json_path=result.get("dom_json_path"),
        error=result.get("error")
    )

@router.get("/download/{job_id}/markdown")
async def download_markdown(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["status"] != "done": raise HTTPException(status_code=404)
    path = job["result"].get("markdown_path")
    if not path or not Path(path).exists(): raise HTTPException(status_code=404)
    return FileResponse(path, media_type="text/markdown", filename=Path(path).name)

@router.get("/download/{job_id}/semantic")
async def download_semantic(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["status"] != "done": raise HTTPException(status_code=404)
    path = job["result"].get("semantic_json_path")
    if not path or not Path(path).exists(): raise HTTPException(status_code=404)
    return FileResponse(path, media_type="application/json", filename=Path(path).name)
