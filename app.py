# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np
import io
import os

from solve_hack import run_full_validation, extract_data_from_image_tesseract

# Optional Supabase integration (guarded)
try:
    from supabase_client import get_supabase  # tiny helper that creates the client from .env
    _SUPABASE_AVAILABLE = True
except Exception:
    _SUPABASE_AVAILABLE = False

app = FastAPI(title="Provider Validation API", version="1.1.0")

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
            if isinstance(v, (float, int)) and (pd.isna(v) or v == float("inf") or v == float("-inf")):
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
    """
    Uploads a CSV file and returns rows for quick preview (no persistence here).
    Matches columns as-is, cleans NaN/Inf for safe JSON.
    """
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
    """Extract text & structured information from uploaded images via Tesseract-based pipeline."""
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
    """
    Runs the provider validation pipeline and persists CSV.
    Summary payload is returned.
    """
    try:
        summary = run_full_validation(input_csv=input_csv, output_csv=output_csv)
        return summary
    except Exception as e:
        return {"error": f"Validation failed: {e}"}


# ---------------------------------------------------------------------
# Results Fetch Endpoints
# ---------------------------------------------------------------------

@app.get("/results")
def results():
    """Return rows from validation_results.csv, cleaning NaN/Inf."""
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"rows": [], "count": 0}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}", "rows": []}

    rows = _df_to_safe_rows(df)
    return {"rows": rows, "count": len(rows)}

@app.get("/results-csv")
def results_csv():
    """Download the latest validation results CSV file."""
    path = "validation_results.csv"
    if not os.path.exists(path):
        return JSONResponse({"detail": "CSV file not found. Please run validation first."}, status_code=404)
    return FileResponse(path, media_type="text/csv", filename="validation_results.csv")

@app.get("/results/summary")
def get_summary():
    """Aggregated counts & avg confidence from validation_results.csv for dashboard KPIs."""
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"total": 0, "verified": 0, "needs_update": 0, "needs_review": 0, "avg_confidence": 0.0}
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna("")
    total = len(df)
    verified = int((df["status"].str.contains("VERIFIED", na=False)).sum()) if "status" in df else 0
    needs_update = int((df["status"].str.contains("UPDATE", na=False)).sum()) if "status" in df else 0
    needs_review = int((df["status"].str.contains("REVIEW", na=False)).sum()) if "status" in df else 0
    avg_conf = float(pd.to_numeric(df.get("confidence_score", pd.Series()), errors="coerce").fillna(0).mean()) if total else 0.0
    return {
        "total": total,
        "verified": verified,
        "needs_update": needs_update,
        "needs_review": needs_review,
        "avg_confidence": round(avg_conf, 1),
    }

@app.get("/results/review")
def review_queue(limit: int = 25):
    """Lowest-confidence providers with status indicating review/manual handling."""
    path = "validation_results.csv"
    if not os.path.exists(path):
        return {"rows": []}
    df = pd.read_csv(path)
    mask = df["status"].str.contains("REVIEW|MANUAL", case=False, na=False) if "status" in df else False
    sub = df[mask].copy()
    if "confidence_score" in sub:
        sub["confidence_score"] = pd.to_numeric(sub["confidence_score"], errors="coerce").fillna(0)
        sub = sub.sort_values("confidence_score").head(limit)
    return {"rows": sub.replace([np.inf, -np.inf], np.nan).fillna("").to_dict(orient="records")}


# ---------------------------------------------------------------------
# Email helper (single template for a provider)
# ---------------------------------------------------------------------

@app.post("/email-template")
def email_template(provider_name: str, current_phone: str = "", suggested_phone: str = "", status: str = ""):
    body = f"""Subject: Quick contact info check — {provider_name}

Hello {provider_name},

We are refreshing our provider directory to ensure members reach you without friction.
Our system flagged your listing as: {status}
Current phone on file: {current_phone or '—'}
Suggested/validated phone: {suggested_phone or '—'}

Could you confirm the correct phone number and address?
Reply with corrections, or call us at 1-800-XXX-XXXX.

Thank you,
Network Operations
"""
    return {"email_body": body}


# ---------------------------------------------------------------------
# Communications: generate emails + summary report
# ---------------------------------------------------------------------

@app.post("/communications/generate")
def comms_generate():
    """
    Generate verification emails + executive summary from validation_results.csv
    and return a small JSON summary (no emojis, deterministic).
    """
    path = "validation_results.csv"
    if not os.path.exists(path):
        return JSONResponse({"error": "Run /run-batch first; validation_results.csv not found"}, status_code=400)

    df = pd.read_csv(path)

    needs_contact = df[
        (df['status'].str.contains('REVIEW|UPDATE', na=False)) |
        (pd.to_numeric(df.get('confidence_score', 0), errors='coerce').fillna(0) < 80)
    ].copy()

    emails = []
    out_dir = "emails_output"
    os.makedirs(out_dir, exist_ok=True)

    for _, p in needs_contact.iterrows():
        issues = []
        phone = str(p.get('phone', ''))
        gphone = str(p.get('google_phone', ''))
        if gphone and gphone != phone:
            issues.append(f"Phone number mismatch (Our records: {phone}, Google Maps: {gphone})")
        try:
            if float(p.get('confidence_score', 0)) < 50:
                issues.append("Multiple data conflicts across sources")
        except Exception:
            pass
        if 'registration_valid' in p and p['registration_valid'] is False:
            issues.append("Medical registration verification required")

        issue_list = "\n".join([f"  • {x}" for x in issues]) if issues else "  • General verification required"
        email_to = (str(p.get('email')) if pd.notna(p.get('email')) else f"{str(p.get('name','')).lower().replace(' ','.')}@hospital.com")

        body = f"""Dear {p.get('name','Provider')},

We are updating our healthcare provider directory to ensure accurate information.

PROVIDER INFORMATION ON FILE
--------------------------------
Name: {p.get('name','N/A')}
Provider ID: {p.get('provider_id','N/A')}
Phone: {p.get('phone','N/A')}
Address: {p.get('address','N/A')}
Specialization: {p.get('specialization','N/A')}

VALIDATION STATUS
--------------------------------
Status: {p.get('status','N/A')}
Confidence Score: {p.get('confidence_score','N/A')}%

ISSUES DETECTED
--------------------------------
{issue_list}

ACTION REQUIRED
--------------------------------
Please confirm or correct the details above. Provide updated:
- Phone
- Address
- Practice Hours
- Specializations

Suggested Correction
--------------------------------
Recommended Phone: {p.get('suggested_phone','N/A')}

Thank you,
Provider Directory Management Team
Reference ID: {p.get('provider_id','')}-{datetime.now().strftime('%Y%m%d')}
"""
        filename = os.path.join(out_dir, f"{p.get('provider_id','UNK')}_{str(p.get('name','')).replace(' ','_')}.txt")
        with open(filename, "w") as f:
            f.write(f"TO: {email_to}\nSUBJECT: Provider Directory Verification Required - {p.get('provider_id','')}\n\n{body}")

        emails.append({
            "provider_id": p.get('provider_id',''),
            "provider_name": p.get('name',''),
            "email_to": email_to,
            "subject": f"Provider Directory Verification Required - {p.get('provider_id','')}",
            "body_path": filename
        })

    emails_csv = "generated_emails.csv"
    pd.DataFrame(emails).to_csv(emails_csv, index=False)

    # compact report
    total = len(df)
    conf = pd.to_numeric(df.get('confidence_score', 0), errors='coerce').fillna(0)
    report_lines = []
    report_lines.append("PROVIDER DIRECTORY VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"Total Providers: {total}")
    report_lines.append(f"Average Confidence: {conf.mean():.1f}%")
    report_lines.append(f"Needs Update: {int(df['status'].str.contains('UPDATE', na=False).sum())}")
    report_lines.append(f"Needs Review: {int(df['status'].str.contains('REVIEW', na=False).sum())}")
    report_lines.append("")
    report_lines.append("Top Priority (lowest confidence):")
    pr = df.copy()
    pr['confidence_score'] = pd.to_numeric(pr.get('confidence_score', 0), errors='coerce').fillna(0)
    pr = pr.sort_values('confidence_score').head(10)
    for _, r in pr.iterrows():
        report_lines.append(f"- {r.get('provider_id','')} | {r.get('name','')} | {r.get('confidence_score',0)}% | {r.get('status','')}")

    report_path = "validation_summary_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    return {
        "emails_count": len(emails),
        "emails_csv": emails_csv,
        "emails_dir": out_dir,
        "report_path": report_path,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/communications/emails-csv")
def comms_emails_csv():
    path = "generated_emails.csv"
    if not os.path.exists(path):
        return JSONResponse({"error": "generated_emails.csv not found. Call /communications/generate first."}, status_code=404)
    return FileResponse(path, media_type="text/csv", filename="generated_emails.csv")

@app.get("/communications/report")
def comms_report():
    path = "validation_summary_report.txt"
    if not os.path.exists(path):
        return JSONResponse({"error": "validation_summary_report.txt not found. Call /communications/generate first."}, status_code=404)
    return FileResponse(path, media_type="text/plain", filename="validation_summary_report.txt")


# ---------------------------------------------------------------------
# Supabase: sync CSV results to DB, and read back
# ---------------------------------------------------------------------

@app.post("/sync/supabase")
def sync_supabase():
    """Push the current validation_results.csv to Supabase manually."""
    if not _SUPABASE_AVAILABLE:
        return JSONResponse({"error": "Supabase not configured"}, status_code=400)

    try:
        path = "validation_results.csv"
        if not os.path.exists(path):
            return JSONResponse({"error": "validation_results.csv not found. Run /run-batch first."}, status_code=400)
        df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)

        total = len(df)
        verified = int((df.get("status", "").str.contains("VERIFIED", na=False)).sum()) if "status" in df else 0
        needs_update = int((df.get("status", "").str.contains("UPDATE", na=False)).sum()) if "status" in df else 0
        needs_review = int((df.get("status", "").str.contains("REVIEW", na=False)).sum()) if "status" in df else 0
        avg_conf = float(pd.to_numeric(df.get("confidence_score", pd.Series()), errors="coerce").fillna(0).mean()) if total else 0.0

        # Reuse push helper from solve_hack if available, otherwise inline here
        try:
            from solve_hack import push_results_to_supabase  # type: ignore
            run_id = push_results_to_supabase(df, {
                "total": total, "verified": verified, "needs_update": needs_update,
                "needs_review": needs_review, "avg_confidence": avg_conf, "output_file": path
            })
        except Exception:
            # Minimal inline writer (latest-run only)
            import uuid
            sb = get_supabase()
            run_id = str(uuid.uuid4())
            sb.table("validation_runs").insert({
                "run_id": run_id,
                "total": total,
                "verified": verified,
                "needs_update": needs_update,
                "needs_review": needs_review,
                "avg_confidence": avg_conf,
                "output_file": path
            }).execute()

            cols = [
                "provider_id","name","registration_number","qualification","specialization",
                "hospital","address","phone","mobile","email","city","state",
                "status","confidence_score","suggested_phone",
                "npi_phone","registration_valid","google_phone","registry_source","source_agreement_count"
            ]
            payload_cols = [c for c in cols if c in df.columns]
            records = df[payload_cols].copy()
            records.insert(0, "run_id", run_id)
            records = records.where(pd.notna(records), None)
            batch = []
            for rec in records.to_dict(orient="records"):
                batch.append(rec)
                if len(batch) >= 500:
                    sb.table("validation_results").upsert(batch).execute()
                    batch = []
            if batch:
                sb.table("validation_results").upsert(batch).execute()

        return {"ok": True, "run_id": run_id, "total": total}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/results/db")
def results_from_db(limit: int = 200, run_id: Optional[str] = None):
    """Fetch rows from Supabase validation_results (latest run if run_id not provided)."""
    if not _SUPABASE_AVAILABLE:
        return JSONResponse({"error": "Supabase not configured"}, status_code=400)

    try:
        sb = get_supabase()
        if not run_id:
            latest = sb.table("validation_runs").select("run_id,started_at").order("started_at", desc=True).limit(1).execute()
            if not latest.data:
                return {"rows": [], "count": 0}
            run_id = latest.data[0]["run_id"]

        data = sb.table("validation_results").select("*").eq("run_id", run_id).limit(limit).execute()
        rows = data.data or []
        for r in rows:
            for k, v in list(r.items()):
                if v is None:
                    r[k] = ""
        return {"run_id": run_id, "rows": rows, "count": len(rows)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/runs")
def list_runs(limit: int = 10):
    """List recent validation runs from Supabase."""
    if not _SUPABASE_AVAILABLE:
        return JSONResponse({"error": "Supabase not configured"}, status_code=400)
    try:
        sb = get_supabase()
        data = sb.table("validation_runs").select("run_id, started_at, total, avg_confidence").order("started_at", desc=True).limit(limit).execute()
        return {"runs": data.data or []}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/results/db/by-run")
def results_by_run(run_id: str, limit: int = 500):
    """Fetch results for a specific run_id from Supabase."""
    if not _SUPABASE_AVAILABLE:
        return JSONResponse({"error": "Supabase not configured"}, status_code=400)
    try:
        sb = get_supabase()
        data = sb.table("validation_results").select("*").eq("run_id", run_id).limit(limit).execute()
        rows = data.data or []
        for r in rows:
            for k, v in list(r.items()):
                if v is None:
                    r[k] = ""
        return {"run_id": run_id, "rows": rows, "count": len(rows)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------
# Providers Seeder (exact CSV schema; rupee fee cleaning)
# ---------------------------------------------------------------------

REQUIRED_PROVIDER_COLS = [
    "provider_id","name","registration_number","qualification","specialization",
    "hospital","address","phone","mobile","email","city","state",
    "consultation_fee","timings","languages","expected_error_type"
]

@app.post("/providers/seed")
def seed_providers(csv_path: str = "input_providers.csv"):
    """
    Seed/Upsert into provider_profiles with columns EXACTLY:
    provider_id,name,registration_number,qualification,specialization,hospital,address,phone,mobile,email,city,state,consultation_fee,timings,languages,expected_error_type
    Sanitizes consultation_fee from rupee-formatted strings to numeric.
    """
    if not _SUPABASE_AVAILABLE:
        return JSONResponse({"error": "Supabase not configured"}, status_code=400)

    if not os.path.exists(csv_path):
        return JSONResponse({"error": f"{csv_path} not found"}, status_code=400)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV: {e}"}, status_code=400)

    # Normalize headers (trim spaces)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_PROVIDER_COLS if c not in df.columns]
    if missing:
        return JSONResponse({"error": f"CSV missing columns: {missing}"}, status_code=400)

    # Keep exact order
    df = df[REQUIRED_PROVIDER_COLS].copy()

    # Sanitize consultation_fee: keep digits and dot only (e.g., "₹1,200 Rs" -> "1200")
    if "consultation_fee" in df.columns:
        df["consultation_fee"] = (
            df["consultation_fee"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .str.replace(r"^\.+$", "", regex=True)
            .replace({"": None})
        )
        df["consultation_fee"] = pd.to_numeric(df["consultation_fee"], errors="coerce")

    df = df.replace({np.nan: None})

    try:
        sb = get_supabase()
        records = df.to_dict(orient="records")
        batch, total = [], 0
        for r in records:
            batch.append(r)
            if len(batch) >= 500:
                sb.table("provider_profiles").upsert(batch).execute()
                total += len(batch)
                batch = []
        if batch:
            sb.table("provider_profiles").upsert(batch).execute()
            total += len(batch)
        return {"ok": True, "upserted": total}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------
# Run Command Reminder
# ---------------------------------------------------------------------

"""
Run this app with:
    uvicorn app:app --reload --port 8000
Then access docs at:
    http://127.0.0.1:8000/docs
"""
