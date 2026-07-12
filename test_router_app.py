"""
test_router_app.py
──────────────────
Minimal FastAPI app to test docx_pipeline_router in isolation.
Run:  python test_router_app.py
Then open: http://localhost:8001/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from docx_pipeline_router import router as docling_router
from markitdown_pipeline_router import router as markitdown_router
from pandoc_pipeline_router import router as pandoc_router

app = FastAPI(title="DOCX Pipeline Router — Test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount both routers side-by-side to compare
app.include_router(docling_router, prefix="/docling", tags=["Docling Pipeline"])
app.include_router(markitdown_router, prefix="/markitdown", tags=["MarkItDown Pipeline"])
app.include_router(pandoc_router, prefix="/pandoc", tags=["Pandoc Pipeline"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_router_app:app", host="0.0.0.0", port=8001, reload=True)
