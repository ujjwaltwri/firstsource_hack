# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import pandas as pd
import io
import os

from solve_hack import run_full_validation, extract_data_from_image_tesseract

app = FastAPI(title="Provider Validation API", version="1.0.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "provider-validation-api"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/routes")
def routes():
    return [{"path": r.path, "methods": list(r.methods)} for r in app.router.routes]

# ----- CSV upload -----
@app.post("/validate-csv")
async def validate_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }

# ----- OCR endpoint -----
@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    content = await file.read()
    tmp_path = "_upload_" + file.filename
    with open(tmp_path, "wb") as f:
        f.write(content)
    result = extract_data_from_image_tesseract(tmp_path)
    return result

# ----- Run batch validation -----
@app.post("/run-batch")
def run_batch(input_csv: Optional[str] = "input_providers.csv",
              output_csv: Optional[str] = "validation_results.csv"):
    summary = run_full_validation(input_csv=input_csv, output_csv=output_csv)
    return summary

# ----- Fetch results -----
@app.get("/results")
def get_results():
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"rows": []}
    df = pd.read_csv(path)
    return {"rows": df.head(500).to_dict(orient="records")}

# ----- Download CSV -----
@app.get("/results-csv")
def results_csv():
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"detail": "CSV not found"}
    return FileResponse(path, media_type="text/csv", filename="validation_results.csv")
