# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import pandas as pd
import numpy as np
import io
import os
from solve_hack import run_full_validation, extract_data_from_image_tesseract

app = FastAPI(title="Provider Validation API", version="1.0.0")

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Utility helper to ensure JSON-safe rows
# ---------------------------------------------------------------------
def _df_to_safe_rows(df: pd.DataFrame) -> list[dict]:
    """Clean DataFrame to be safely JSON serializable."""
    if df.empty:
        return []

    # Replace infinities and NaN with clean defaults
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")

    # Round numeric confidence values
    if "confidence_score" in df.columns:
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0).round(1)

    # Convert everything to plain Python objects
    rows = []
    for _, row in df.iterrows():
        safe_row = {}
        for k, v in row.items():
            if isinstance(v, (float, int)) and (pd.isna(v) or np.isinf(v)):
                safe_row[k] = 0
            elif pd.isna(v):
                safe_row[k] = ""
            else:
                safe_row[k] = v
        rows.append(safe_row)

    return rows


# ---------------------------------------------------------------------
# Core Routes
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "provider-validation-api"}

@app.get("/health")
def health():
    return {"ok": True, "message": "Server is healthy"}

@app.get("/routes")
def routes():
    """List all available routes."""
    return [{"path": r.path, "methods": list(r.methods)} for r in app.router.routes]


# ---------------------------------------------------------------------
# CSV Upload and Quick Validation
# ---------------------------------------------------------------------

@app.post("/validate-csv")
async def validate_csv(file: UploadFile = File(...)):
    """Uploads a CSV file and performs quick structure validation."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return {"error": f"Failed to parse CSV: {e}", "rows": []}

    rows = _df_to_safe_rows(df)
    return {
        "rows": rows,
        "columns": list(df.columns),
        "count": len(rows),
    }


# ---------------------------------------------------------------------
# OCR Endpoint
# ---------------------------------------------------------------------

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """Extract text & structured information from uploaded images."""
    try:
        contents = await file.read()
        tmp_path = f"_upload_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(contents)

        result = extract_data_from_image_tesseract(tmp_path)

        # Remove temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return result
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------
# Batch Validation Runner
# ---------------------------------------------------------------------

@app.post("/run-batch")
def run_batch(
    input_csv: Optional[str] = "input_providers.csv",
    output_csv: Optional[str] = "validation_results.csv"
):
    """Runs the provider validation pipeline."""
    try:
        summary = run_full_validation()
    except Exception as e:
        return {"error": f"Validation failed: {e}"}

    return summary


# ---------------------------------------------------------------------
# Results Fetch Endpoint
# ---------------------------------------------------------------------

@app.get("/results")
def results():
    """Safely return rows from validation_results.csv, cleaning NaN/Inf."""
    path = "validation_results.csv"

    if not os.path.exists(path):
        return {"rows": []}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}", "rows": []}

    rows = _df_to_safe_rows(df)
    return {"rows": rows, "count": len(rows)}


# ---------------------------------------------------------------------
# Download CSV
# ---------------------------------------------------------------------

@app.get("/results-csv")
def results_csv():
    """Allows downloading of the latest validation results."""
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"detail": "CSV file not found. Please run validation first."}
    return FileResponse(path, media_type="text/csv", filename="validation_results.csv")


# ---------------------------------------------------------------------
# Run Command Reminder
# ---------------------------------------------------------------------

"""
Run this app with:
    uvicorn app:app --reload --port 8000
Then access docs at:
    http://127.0.0.1:8000/docs
"""

