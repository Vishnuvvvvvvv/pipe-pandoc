"""
markitdown_pipeline_router.py
─────────────────────────────
Self-contained FastAPI router — drop this single file into any FastAPI project.
Uses Microsoft MarkItDown for Markdown extraction and PyPandoc for DOM extraction.

USAGE in your existing main.py:
    from markitdown_pipeline_router import router as docx_router
    app.include_router(docx_router, prefix="/docx", tags=["DOCX Pipeline"])
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
import re as _re
import time as _time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

try:
    from markitdown import MarkItDown
except ImportError:
    pass # Handled at runtime or ensure installed

try:
    import pypandoc
except ImportError:
    pass

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("markitdown_pipeline")

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

# ══════════════════════════════════════════════════════════════════════════════
# AWS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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
        lines = [
            b["Text"] for b in response.get("Blocks", [])
            if b.get("BlockType") == "LINE"
        ]
        return "\n".join(lines).strip()
    except (BotoCoreError, ClientError) as exc:
        log.warning("Textract OCR failed: %s", exc)
        return ""
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
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                    {"text": _CHART_PROMPT},
                ],
            }],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        parsed.setdefault("chart_type", "unknown")
        parsed.setdefault("series", [])
        parsed.setdefault("summary", "")
        return parsed
    except (BotoCoreError, ClientError) as exc:
        log.error("Bedrock call failed: %s", exc)
        return {**_ERROR_RESULT, "summary": str(exc)}
    except json.JSONDecodeError as exc:
        log.warning("Bedrock non-JSON response: %s", exc)
        return {**_ERROR_RESULT, "summary": f"JSON parse error: {exc}"}
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
    
    out = []
    for i, t_node in enumerate(raw_tables):
        out.append({
            "table_index": i,
            "raw_pandoc_ast": t_node, 
            # We don't try to build a flattened DataFrame here since Pandoc AST is extremely nested,
            # but we preserve it in semantic.json for layout tasks.
        })
    return out

def process_extracted_images(image_dir: Path) -> list[dict[str, Any]]:
    """Process images extracted by pypandoc."""
    if not image_dir.exists():
        return []
    
    out = []
    idx = 0
    for img_path in sorted(image_dir.rglob("*")):
        if not img_path.is_file():
            continue
        try:
            with Image.open(img_path) as img:
                img_copy = img.copy()
            
            ocr_text = run_ocr(img_copy)
            
            if USE_VLM and looks_like_chart(img_copy, ocr_text):
                semantic = describe_chart(img_path)
            else:
                semantic = _NOT_EVALUATED.copy()
                
            out.append({
                "picture_index": idx,
                "file": str(img_path),
                "width": img_copy.width,
                "height": img_copy.height,
                "ocr_text": ocr_text,
                "semantic": semantic,
                "resolution_method": "pypandoc",
            })
            idx += 1
        except Exception as exc:
            log.warning("Could not process image %s: %s", img_path.name, exc)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN RENDERER (Post-Processing)
# ══════════════════════════════════════════════════════════════════════════════

def render_markdown(base_md: str, tables: list, images: list, docx_path: Optional[Path] = None) -> str:
    # We keep the post-processing lightweight since MarkItDown already produces very clean markdown,
    # but we will append the semantic image appendix as before.
    
    parts = [base_md]

    if images:
        parts.append("\n\n---\n## Extracted Image & Chart Data\n")
        for rec in images:
            fname = Path(rec["file"]).name
            parts.append(f"\n### Image {rec['picture_index']}  (`{fname}`)")
            if rec.get("caption"):
                parts.append(f"**Caption:** {rec['caption']}")
            parts.append(f"**Dimensions:** {rec.get('width','?')}×{rec.get('height','?')} px")
            if rec.get("ocr_text"):
                parts.append(f"\n**OCR text:**\n```\n{rec['ocr_text']}\n```")
            sem = rec.get("semantic") or {}
            ct = sem.get("chart_type", "")
            if ct and ct not in ("not_evaluated", "not_a_chart", "unavailable", "error", ""):
                parts.append(f"\n**Chart type:** `{ct}`")
                if sem.get("summary"):
                    parts.append(f"**Summary:** {sem['summary']}")
                for pt in sem.get("series", []):
                    parts.append(f"- {pt.get('label','')}: {pt.get('value','')}")

    return "\n".join(parts)


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _process_document(docx_path: Path) -> dict[str, Any]:
    name = docx_path.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "images"
    media_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing with MarkItDown & PyPandoc: %s", docx_path.name)

    # Pre-process: fix ListBullet2/ListNumber2 style nesting
    # (same pre-processor used by the Pandoc pipeline)
    try:
        from pandoc_pipeline_router import _preprocess_list_nesting
        fixed_path = _preprocess_list_nesting(docx_path)
    except Exception:
        fixed_path = docx_path

    # 1. Generate Markdown using MarkItDown
    try:
        md_converter = MarkItDown()
        result = md_converter.convert(str(fixed_path))
        base_md = result.text_content
    except Exception as exc:
        log.error("MarkItDown failed: %s", exc)
        return {"success": False, "document": docx_path.name, "error": f"MarkItDown error: {exc}",
                "markdown": "", "tables_count": 0, "images_count": 0}
    finally:
        if fixed_path != docx_path:
            try:
                fixed_path.unlink(missing_ok=True)
            except Exception:
                pass

    # 2. Extract DOM (AST) and media
    dom_ast = {}
    try:
        dom_json_str = pypandoc.convert_file(str(docx_path), 'json')
        dom_ast = json.loads(dom_json_str)
    except Exception as exc:
        log.warning("PyPandoc DOM extraction failed: %s", exc)

    import zipfile
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx_zip:
            media_files = [n for n in docx_zip.namelist() if n.startswith("word/media/")]
            for m in media_files:
                if not m.endswith('/'):  # skip directories
                    img_data = docx_zip.read(m)
                    dest = media_dir / Path(m).name
                    dest.write_bytes(img_data)
    except Exception as exc:
        log.warning("Zipfile media extraction failed: %s", exc)

    # Write DOM
    dom_path = out_dir / f"{name}.dom.json"
    try:
        dom_path.write_text(
            json.dumps(dom_ast, indent=2, ensure_ascii=False, cls=_SafeEncoder),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("DOM export failed: %s", exc)

    # 3. Process Tables + Images
    tables = extract_tables_from_ast(dom_ast) if dom_ast else []
    images = process_extracted_images(media_dir)

    # 4. Semantic JSON
    semantic_path = out_dir / f"{name}.semantic.json"
    semantic: dict[str, Any] = {
        "document":          docx_path.name,
        "schema_version":    "1.0",
        "num_pages":         None,
        "text_blocks_count": len(dom_ast.get("blocks", [])) if dom_ast else 0,
        "tables":            tables,
        "images":            images,
    }
    semantic_path.write_text(
        json.dumps(semantic, indent=2, ensure_ascii=False, cls=_SafeEncoder),
        encoding="utf-8",
    )

    # 5. Render Final Markdown
    full_md = render_markdown(base_md, tables, images, docx_path=docx_path)
    md_path = out_dir / f"{name}.md"
    md_path.write_text(full_md, encoding="utf-8")

    log.info("Done: %s  (tables=%d, images=%d)", docx_path.name, len(tables), len(images))

    return {
        "success":            True,
        "document":           docx_path.name,
        "error":              None,
        "markdown":           full_md,
        "dom_json_path":      str(dom_path),
        "semantic_json_path": str(semantic_path),
        "markdown_path":      str(md_path),
        "tables_count":       len(tables),
        "images_count":       len(images),
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
            "result": {"success": False, "error": str(exc),
                       "markdown": "", "tables_count": 0, "images_count": 0},
        })
    finally:
        try:
            docx_path.unlink(missing_ok=True)
        except Exception:
            pass


def _safe_unlink(path: Path, retries: int = 3, delay: float = 0.5) -> None:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < retries - 1:
                _time.sleep(delay)
            else:
                log.warning("Could not delete temp file '%s'", path.name)
        except Exception:
            return


def _convert_doc_to_docx(doc_path: Path) -> Optional[Path]:
    out_path = STAGING_DIR / (doc_path.stem + ".docx")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from doc2docx import convert
        convert(str(doc_path), str(out_path))
        if out_path.exists():
            return out_path
    except Exception:
        pass

    import subprocess
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        try:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "docx",
                 "--outdir", str(STAGING_DIR), str(doc_path)],
                check=True, capture_output=True, timeout=120,
            )
            if out_path.exists():
                return out_path
        except Exception:
            pass

    return None

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Only .doc and .docx accepted. Got: '{suffix}'",
        )

    job_id = str(uuid.uuid4())
    tmp_path = STAGING_DIR / f"{job_id}{suffix}"

    try:
        with tmp_path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()

    if suffix == ".doc":
        converted = _convert_doc_to_docx(tmp_path)
        _safe_unlink(tmp_path)
        if converted is None:
            raise HTTPException(status_code=422, detail="Could not convert .doc")
        tmp_path = converted

    _jobs[job_id] = {
        "status":   "queued",
        "result":   None,
        "filename": file.filename,
    }

    background_tasks.add_task(_executor.submit, _run_pipeline, job_id, tmp_path)

    return UploadResponse(
        job_id=job_id,
        status="queued",
        message=f"'{file.filename}' queued. Poll /status/{job_id}",
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    result: dict = job.get("result") or {}
    md = result.get("markdown", "")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        document=result.get("document") or job.get("filename"),
        tables_count=result.get("tables_count"),
        images_count=result.get("images_count"),
        markdown_preview=md[:2000] if md else None,
        markdown_path=result.get("markdown_path"),
        semantic_json_path=result.get("semantic_json_path"),
        dom_json_path=result.get("dom_json_path"),
        error=result.get("error"),
    )

@router.get("/download/{job_id}/markdown")
async def download_markdown(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete.")
    md_path = job["result"].get("markdown_path")
    if not md_path or not Path(md_path).exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(md_path, media_type="text/markdown", filename=Path(md_path).name)

@router.get("/download/{job_id}/semantic")
async def download_semantic(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete.")
    path = job["result"].get("semantic_json_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="application/json", filename=Path(path).name)
