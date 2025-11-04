# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
from communications import generate_all, generate_verification_emails, generate_summary_report

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

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), save_as: Optional[str] = "input_providers.csv"):
    """Save an uploaded CSV to disk so /run-batch can process it."""
    content = await file.read()
    try:
        pd.read_csv(io.BytesIO(content))  # validate parse
    except Exception as e:
        return {"error": f"Failed to parse CSV: {e}"}

    save_path = Path(save_as).as_posix()
    with open(save_path, "wb") as f:
        f.write(content)
    return {"saved_as": save_path, "ok": True}


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

        # Add a preview slice to keep payloads light for the UI
        if isinstance(result, dict) and "lines_with_confidence" in result:
            lines = result["lines_with_confidence"]
            result["lines_with_confidence_preview"] = lines[:200] if isinstance(lines, list) else []

        # Remove temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return result
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------
# Batch Validation Runner (legacy alias to avoid duplicate routes)
# ---------------------------------------------------------------------

@app.post("/run-batch-simple")
def run_batch(
    input_csv: Optional[str] = "input_providers.csv",
    output_csv: Optional[str] = "validation_results.csv"
):
    """Legacy alias that runs with defaults (kept for backwards compat)."""
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
# Enhanced Batch Runner (forwards filenames)
# ---------------------------------------------------------------------

@app.post("/run-batch")
def run_batch_forwarding(
    input_csv: Optional[str] = "input_providers.csv",
    output_csv: Optional[str] = "validation_results.csv"
):
    """Runs the provider validation pipeline with file arguments forwarded."""
    try:
        summary = run_full_validation(input_csv=input_csv, output_csv=output_csv)
        return summary
    except Exception as e:
        return {"error": f"Validation failed: {e}"}


# ---------------------------------------------------------------------
# Summary + Review Queue + Email Template
# ---------------------------------------------------------------------

@app.get("/results/summary")
def get_summary():
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"total":0,"verified":0,"needs_update":0,"needs_review":0,"avg_confidence":0.0,"elapsed_seconds":0.0}
    df = pd.read_csv(path).replace([np.inf,-np.inf], np.nan).fillna("")
    total = len(df)
    verified = int((df["status"].str.contains("VERIFIED", na=False)).sum()) if "status" in df else 0
    needs_update = int((df["status"].str.contains("UPDATE", na=False)).sum()) if "status" in df else 0
    needs_review = int((df["status"].str.contains("REVIEW", na=False)).sum()) if "status" in df else 0
    avg_conf = float(pd.to_numeric(df.get("confidence_score", pd.Series()), errors="coerce").fillna(0).mean()) if total else 0.0
    return {"total": total, "verified": verified, "needs_update": needs_update, "needs_review": needs_review,
            "avg_confidence": round(avg_conf,1), "elapsed_seconds": 0.0}

@app.post("/email-template")
def email_template(provider_name: str, current_phone: str = "", suggested_phone: str = "", status: str = ""):
    body = f"""Subject: Quick contact info check — {provider_name}

Hello {provider_name},

We’re refreshing our provider directory to ensure members reach you without friction.
Our system flagged your listing as: {status}
Current phone on file: {current_phone or '—'}
Suggested/validated phone: {suggested_phone or '—'}

Could you confirm the correct phone number and address? 
Reply with corrections, or call us at 1-800-XXX-XXXX.

Thank you,
Network Operations
"""
    return {"email_body": body}

@app.get("/results/review")
def review_queue(limit: int = 25):
    path="validation_results.csv"
    if not os.path.exists(path): 
        return {"rows":[]}
    df = pd.read_csv(path)
    mask = df["status"].str.contains("REVIEW|MANUAL", case=False, na=False) if "status" in df else False
    sub = df[mask].copy()
    if "confidence_score" in sub:
        sub["confidence_score"] = pd.to_numeric(sub["confidence_score"], errors="coerce").fillna(0)
        sub = sub.sort_values("confidence_score").head(limit)
    return {"rows": sub.replace([np.inf,-np.inf], np.nan).fillna("").to_dict(orient="records")}


# ---------------------------------------------------------------------
# Run Command Reminder
# ---------------------------------------------------------------------
"""
Run this app with:
    uvicorn app:app --reload --port 8000
Then access docs at:
    http://127.0.0.1:8000/docs
"""
# ---------------------------------------------------------------------
# Communications: generate emails + summary report
# ---------------------------------------------------------------------

@app.post("/communications/generate")
def communications_generate(background: BackgroundTasks):
    """
    Generate verification emails (CSV + individual .txt) and the summary report.
    Runs inline (fast on 200 rows). Returns file paths and counts.
    """
    try:
        summary = generate_all()
        return JSONResponse(summary)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/communications/emails-csv")
def communications_emails_csv():
    """
    Download the generated_emails.csv (call /communications/generate first).
    """
    path = "generated_emails.csv"
    if not os.path.exists(path):
        return JSONResponse({"detail": "generated_emails.csv not found. Run /communications/generate first."}, status_code=404)
    return FileResponse(path, media_type="text/csv", filename="generated_emails.csv")


@app.get("/communications/report")
def communications_report():
    """
    Download the validation_summary_report.txt (call /communications/generate first).
    """
    path = "validation_summary_report.txt"
    if not os.path.exists(path):
        return JSONResponse({"detail": "validation_summary_report.txt not found. Run /communications/generate first."}, status_code=404)
    return FileResponse(path, media_type="text/plain", filename="validation_summary_report.txt")
