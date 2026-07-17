"""kk
pandoc_app.py
─────────────
Lean FastAPI app — mounts ONLY the pandoc pipeline router.
Used for Docker deployment (no PyTorch/CUDA/Docling required).

Run:
    uvicorn pandoc_app:app --host 0.0.0.0 --port 8001
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pandoc_pipeline_router import router as pandoc_router

app = FastAPI(
    title="DOCX → Markdown Pipeline (Pandoc)",
    description="Converts .doc/.docx files to Markdown using Pandoc.",
    version="1.0.0",
)
//cors exten's
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pandoc_router, prefix="/pandoc", tags=["Pandoc Pipeline"])

@app.get("/health")
def health():
    return {"status": "ok", "engine": "pandoc"}
