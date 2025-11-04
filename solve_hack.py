# solve_hack.py
"""
Provider Validation & OCR pipeline for Firstsource hackathon.
- Network-safe wrappers for NPI / Google Map calls
- Optional parallel validation (USE_PARALLEL=1)
- Tesseract OCR (multiple preprocessing pipelines)
- Change log generation and PII-safe logs
"""
import uuid
from supabase_client import get_supabase
import os
import time
import json
import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import pytesseract
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Try enable AVIF support (optional)
try:
    from pillow_avif import AvifImagePlugin  # noqa: F401
    logging.info("AVIF support enabled")
except Exception:
    logging.info("AVIF support not available (pip install pillow-avif-plugin to enable)")

# Load env
load_dotenv()

# ----------------------
# Utilities
# ----------------------
def _mask_phone(p: Optional[str]) -> str:
    s = "".join([c for c in (str(p) if p is not None else "") if c.isdigit()])
    if len(s) >= 4:
        return ("*" * max(0, len(s) - 4)) + s[-4:]
    return "***"

def _safe_print_provider_row(row) -> None:
    try:
        logging.info(f"[{row.get('provider_id', '?')}] {row.get('name','?')} — {_mask_phone(row.get('phone',''))}")
    except Exception:
        pass

def _get_with_retry(url: str, params: dict = None, timeout: int = 5, retries: int = 2, backoff: float = 0.5):
    """GET wrapper with exponential backoff retry. Raises on final failure."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries:
                logging.debug(f"_get_with_retry final failure: {e} (url={url})")
                raise
            sleep_for = backoff * (2 ** attempt)
            logging.debug(f"_get_with_retry retry {attempt+1}/{retries} sleeping {sleep_for}s")
            time.sleep(sleep_for)

# ----------------------
# NPI Registry (US)
# ----------------------
def get_npi_data(npi_number: str):
    """Returns dict with npi_phone and npi_name or None."""
    if pd.isna(npi_number) or str(npi_number).strip() == "":
        return None
    url = f"https://npiregistry.cms.hhs.gov/api/"
    params = {"number": str(npi_number).strip(), "version": "2.1"}
    try:
        resp = _get_with_retry(url, params=params, timeout=6, retries=2)
        data = resp.json()
        if data.get("result_count", 0) == 0:
            return None
        provider = data["results"][0]
        practice_locations = [addr for addr in provider.get("addresses", []) if addr.get("address_purpose") == "LOCATION"]
        clean_phone = None
        for loc in practice_locations:
            raw_phone = loc.get("telephone_number", "")
            if raw_phone:
                clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                break
        if not clean_phone:
            for addr in provider.get("addresses", []):
                raw_phone = addr.get("telephone_number", "")
                if raw_phone:
                    clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                    break
        basic_info = provider.get("basic", {})
        if basic_info.get("first_name"):
            npi_name = f"{basic_info.get('first_name','')} {basic_info.get('last_name','')}".strip()
        else:
            npi_name = basic_info.get("organization_name")
        return {"npi_phone": clean_phone, "npi_name": npi_name}
    except Exception as e:
        logging.debug(f"get_npi_data error for {npi_number}: {e}")
        return None

# ----------------------
# Indian Medical Council mock validator
# ----------------------
def get_indian_medical_council_data(registration_number: str):
    """Mock validation for Indian provider registration numbers."""
    if not registration_number or str(registration_number).strip() == "":
        return {"registration_valid": False, "error": "Blank"}
    pattern = r'^[A-Z]{2,4}/\d{4,5}/\d{4}$'
    if not re.match(pattern, str(registration_number)):
        return {"registration_valid": False, "error": "Invalid format"}
    parts = str(registration_number).split('/')
    state_code = parts[0]
    year = parts[2]
    try:
        year_int = int(year)
        if year_int < 1900 or year_int > time.localtime().tm_year:
            return {"registration_valid": False, "error": "Invalid registration year"}
    except Exception:
        return {"registration_valid": False, "error": "Invalid registration year"}
    # Demo randomness to mimic lookup result
    import random
    if random.random() < 0.8:
        return {
            "registration_valid": True,
            "state_code": state_code,
            "registration_year": year,
            "status": "Active",
            "council_name": f"{state_code} Medical Council"
        }
    else:
        return {"registration_valid": False, "error": "Not found"}

# ----------------------
# Google Maps Places (optional)
# ----------------------
def get_google_maps_data(provider_name: str, provider_address: str):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        logging.debug("No Google Maps API key; skipping maps lookup.")
        return None
    search_query = f"{provider_name or ''} {provider_address or ''}".strip()
    if not search_query:
        return None
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    try:
        resp = _get_with_retry(search_url, params={"query": search_query, "key": api_key}, timeout=6, retries=2)
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            place_id = data["results"][0]["place_id"]
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_resp = _get_with_retry(details_url, params={"place_id": place_id, "fields": "name,formatted_phone_number", "key": api_key}, timeout=6, retries=2)
            details = details_resp.json()
            if details.get("status") == "OK" and details.get("result"):
                phone = details["result"].get("formatted_phone_number", "")
                clean_phone = "".join(filter(str.isdigit, phone))
                if len(clean_phone) > 10:
                    clean_phone = clean_phone[-10:]
                return {"google_phone": clean_phone if clean_phone else None}
    except Exception as e:
        logging.debug(f"get_google_maps_data error ({search_query}): {e}")
    return None

# ----------------------
# Validation per-row
# ----------------------
def validate_provider_row(row):
    """
    Accepts a pandas Series (row), returns a dict containing status, confidence, suggested_phone etc.
    """
    try:
        input_phone = str(row.get("phone", "") or "")
        input_phone_clean = "".join(filter(str.isdigit, input_phone))
        if len(input_phone_clean) > 10:
            input_phone_clean = input_phone_clean[-10:]

        has_npi = "npi_number" in row and pd.notna(row.get("npi_number")) and str(row.get("npi_number")).strip() != ""
        has_indian_reg = "registration_number" in row and pd.notna(row.get("registration_number")) and str(row.get("registration_number")).strip() != ""

        npi_phone = None
        registration_valid = False
        registry_source = None

        if has_npi:
            npi_data = get_npi_data(str(row.get("npi_number")))
            npi_phone = npi_data.get("npi_phone") if npi_data else None
            registry_source = "NPI Registry"
        elif has_indian_reg:
            reg_data = get_indian_medical_council_data(str(row.get("registration_number")))
            registration_valid = bool(reg_data.get("registration_valid", False))
            if registration_valid:
                registry_source = reg_data.get("council_name", "State Medical Council")

        google_data = get_google_maps_data(row.get("name", ""), row.get("address", ""))
        google_phone = google_data.get("google_phone") if google_data else None

        # Status / confidence logic
        status = "NEEDS_MANUAL_REVIEW"
        confidence = 30
        suggested = None

        if google_phone and input_phone_clean == google_phone:
            if npi_phone == google_phone or registration_valid:
                status = f"VERIFIED_OK (Google + {registry_source})"
                confidence = 100
                suggested = google_phone
            else:
                status = "VERIFIED_OK (Google only)"
                confidence = 85
                suggested = google_phone
        elif google_phone and input_phone_clean != google_phone:
            status = "NEEDS_UPDATE (Google)"
            confidence = 90
            suggested = google_phone
        elif npi_phone:
            if input_phone_clean == npi_phone:
                status = "VERIFIED_OK (NPI)"
                confidence = 80
                suggested = npi_phone
            else:
                status = "NEEDS_UPDATE (NPI)"
                confidence = 70
                suggested = npi_phone
        elif registration_valid:
            status = f"NEEDS_REVIEW ({registry_source})"
            confidence = 60
            suggested = input_phone_clean if input_phone_clean else None

        sources = [p for p in [input_phone_clean or None, google_phone, npi_phone] if p]
        source_agreement_count = max((sources.count(ph) for ph in set(sources)), default=0)

        return {
            "status": status,
            "confidence_score": int(confidence),
            "suggested_phone": suggested,
            "npi_phone": npi_phone if has_npi else None,
            "registration_valid": bool(registration_valid) if has_indian_reg else None,
            "google_phone": google_phone,
            "registry_source": registry_source,
            "source_agreement_count": int(source_agreement_count),
        }
    except Exception as e:
        logging.debug(f"validate_provider_row error: {e}")
        return {
            "status": "NEEDS_MANUAL_REVIEW",
            "confidence_score": 20,
            "suggested_phone": None,
            "error": str(e)
        }

# ----------------------
# Optional parallel helper
# ----------------------
def _validate_rows_parallel_ordered(df: pd.DataFrame, max_workers: int = 16):
    """Run validate_provider_row in parallel while preserving original order."""
    rows = [row for _, row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(validate_provider_row, rows))
    return results

# ----------------------
# OCR helpers
# ----------------------
def convert_image_to_supported_format(image_path: str):
    """Convert image to PNG using PIL; fallback to OpenCV if needed."""
    logging.debug(f"Converting image: {image_path}")
    try:
        img = Image.open(image_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        temp_path = "temp_converted.png"
        img.save(temp_path, "PNG")
        return temp_path
    except Exception as e:
        logging.debug(f"PIL conversion failed: {e}, trying OpenCV")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to open image for conversion: {image_path}")
        temp_path = "temp_converted.png"
        cv2.imwrite(temp_path, img)
        return temp_path

def enhance_image_for_ocr(image_path: str):
    logging.debug("Enhancing image for OCR")
    file_ext = Path(image_path).suffix.lower()
    working_path = image_path
    if file_ext in [".avif", ".webp", ".heic"]:
        working_path = convert_image_to_supported_format(image_path)
    img = cv2.imread(working_path)
    if img is None:
        raise ValueError(f"Could not read image: {working_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = []
    # Otsu
    th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed.append(("otsu", th1))
    # Adaptive
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    processed.append(("adaptive", th2))
    # Denoise + Otsu
    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    th3 = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed.append(("denoised", th3))
    # CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    th4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed.append(("clahe", th4))
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    processed.append(("morph", morph))
    # Bilateral + Otsu
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    th5 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed.append(("bilateral", th5))
    return processed

def parse_medical_pamphlet_advanced(text: str):
    info = {
        "doctor_name": None,
        "credentials": None,
        "hospital_name": None,
        "specialization": None,
        "phone_numbers": [],
        "email": None,
        "website": None,
        "conditions_treated": []
    }
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        up = line.upper()
        # doctor name heuristics
        if (up.startswith("DR.") or up.startswith("DR ")) and not info["doctor_name"]:
            candidate = re.sub(r'[|\\\/]', '', line).strip()
            info["doctor_name"] = re.sub(r'\s+', ' ', candidate)
            if i + 1 < len(lines):
                nxt = lines[i + 1].upper()
                if any(x in nxt for x in ["MBBS", "MD", "DLO", "MS"]):
                    info["credentials"] = lines[i + 1].strip()
        # hospital
        if "HOSPITAL" in up and not info["hospital_name"]:
            clean = re.sub(r'[^A-Za-z\s]', '', line)
            info["hospital_name"] = re.sub(r'\s+', ' ', clean).strip()
        # phone
        phone_match = re.search(r'\b\d{4,5}[-\s]?\d{6,8}\b', line)
        if phone_match:
            p = "".join(filter(str.isdigit, phone_match.group()))
            if p not in info["phone_numbers"]:
                info["phone_numbers"].append(p)
        # email
        email_match = re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', line)
        if email_match and not info["email"]:
            info["email"] = email_match.group().lower()
        # website
        web_match = re.search(r'(https?://\S+|www\.\S+)', line, re.IGNORECASE)
        if web_match and not info["website"]:
            info["website"] = web_match.group().lower()
        # specialization heuristics
        if any(term in up for term in ["SURGEON", "SPECIALIST", "CARDIO", "ENT", "PEDIATRIC", "ORTHO"]) and not info["specialization"]:
            clean = re.sub(r'[^A-Za-z\s,&]', '', line)
            info["specialization"] = re.sub(r'\s+', ' ', clean).strip()
    # conditions
    medical_terms = {'EAR','THROAT','TONSIL','ALLERG','NOSE','HEARING','VERTIGO','REFLUX','THYROID','PAIN','INFECTION'}
    for line in lines:
        up = line.upper()
        if any(t in up for t in medical_terms) and not any(skip in up for skip in ['CONTACT','CALL','DR.']):
            cleaned = re.sub(r'[^A-Za-z\s&]', '', line).strip()
            if 5 < len(cleaned) < 60 and cleaned not in info["conditions_treated"]:
                info["conditions_treated"].append(cleaned)
    return info

def extract_data_from_image_tesseract(image_path: str):
    logging.info(f"Starting OCR: {image_path}")
    try:
        file_ext = Path(image_path).suffix.lower()
        working_path = image_path
        if file_ext in [".avif", ".webp", ".heic"]:
            working_path = convert_image_to_supported_format(image_path)
        processed = enhance_image_for_ocr(working_path)
        all_results = []
        # original
        original_img = Image.open(working_path)
        cfg = r'--oem 3 --psm 6'
        original_text = pytesseract.image_to_string(original_img, config=cfg)
        all_results.append(("original", original_text, len(original_text.strip())))
        # processed variants
        for name, img in processed:
            text = pytesseract.image_to_string(img, config=cfg)
            all_results.append((name, text, len(text.strip())))
        best = max(all_results, key=lambda x: x[2])
        best_name, best_text, char_count = best
        parsed = parse_medical_pamphlet_advanced(best_text)
        # line-level confidence (use first processed if available)
        sample_img = processed[0][1] if processed else None
        detailed = {}
        try:
            detailed = pytesseract.image_to_data(sample_img if sample_img is not None else original_img, output_type=pytesseract.Output.DICT, config=cfg)
        except Exception:
            detailed = {}
        lines_with_confidence = []
        if detailed and "text" in detailed:
            for i, t in enumerate(detailed.get("text", [])):
                if str(t).strip():
                    conf = detailed.get("conf", [])[i] if i < len(detailed.get("conf", [])) else None
                    lines_with_confidence.append({"text": str(t).strip(), "confidence": conf})
        result = {
            "model_used": "Tesseract OCR",
            "best_preprocessing_method": best_name,
            "total_characters_extracted": char_count,
            "full_text": best_text,
            "lines_with_confidence": lines_with_confidence,
            "parsed_information": parsed,
            "all_methods_tested": [r[0] for r in all_results]
        }
        # cleanup temp files
        for t in ("temp_converted.png", "temp_preprocessed.jpg"):
            if os.path.exists(t):
                try:
                    os.remove(t)
                except Exception:
                    pass
        return result
    except Exception as e:
        logging.exception("OCR pipeline failed")
        return {"error": str(e)}

# ----------------------
# Orchestrator
# ----------------------
def run_full_validation(input_csv: str = "input_providers.csv", output_csv: str = "validation_results.csv"):
    start_ts = time.time()
    if not os.path.exists(input_csv):
        logging.error(f"Input CSV not found: {input_csv}")
        return {"error": f"{input_csv} not found"}
    df = pd.read_csv(input_csv, dtype=str).fillna("")
    logging.info(f"Loaded {len(df)} providers from {input_csv}")

    use_parallel = os.getenv("USE_PARALLEL", "0") == "1"
    max_workers = int(os.getenv("MAX_WORKERS", "12"))
    logging.info(f"Validation mode: {'parallel' if use_parallel else 'serial'} (workers={max_workers})")

    if use_parallel:
        validation_results = _validate_rows_parallel_ordered(df, max_workers=max_workers)
        validation_series = pd.Series(validation_results)
    else:
        validation_series = df.apply(validate_provider_row, axis=1)

    final_df = pd.concat([df.reset_index(drop=True), validation_series.apply(pd.Series).reset_index(drop=True)], axis=1)

    # drop helper column if present
    if "expected_error_type" in final_df.columns:
        final_df = final_df.drop(columns=["expected_error_type"], errors="ignore")

    final_df.to_csv(output_csv, index=False)
    elapsed = round(time.time() - start_ts, 2)

    total = len(final_df)
    verified = int(final_df['status'].str.contains('VERIFIED', na=False).sum()) if 'status' in final_df else 0
    try:
        needs_update = int(final_df['status'].str.contains('UPDATE', na=False).sum()) if 'status' in final_df else 0
    except Exception:
        needs_update = 0
    needs_review = int(final_df['status'].str.contains('REVIEW', na=False).sum()) if 'status' in final_df else 0
    avg_conf = float(pd.to_numeric(final_df.get("confidence_score", pd.Series()), errors="coerce").fillna(0).mean()) if total else 0.0

    logging.info("VALIDATION COMPLETE")
    logging.info(f"Total: {total} Verified: {verified} NeedsUpdate: {needs_update} NeedsReview: {needs_review} AvgConf: {avg_conf:.1f} Elapsed: {elapsed}s")
    # write change log for suggestions
    try:
        with open("change_log.jsonl", "a") as lf:
            for _, r in final_df.iterrows():
                try:
                    if str(r.get("status","")).startswith("NEEDS_") and r.get("suggested_phone"):
                        lf.write(json.dumps({
                            "timestamp": int(time.time()),
                            "provider_id": r.get("provider_id"),
                            "name": r.get("name"),
                            "current_phone": r.get("phone"),
                            "suggested_phone": r.get("suggested_phone"),
                            "source": r.get("registry_source") or ("Google" if r.get("google_phone") else None),
                            "status": r.get("status"),
                            "confidence": r.get("confidence_score")
                        }) + "\n")
                except Exception:
                    continue
    except Exception as e:
        logging.debug(f"Failed to write change_log.jsonl: {e}")

    # Print top 10 rows for quick CLI inspection (masked)
    try:
        display_cols = [c for c in ['provider_id','name','status','confidence_score','suggested_phone'] if c in final_df.columns]
        logging.info("Top 10 results (masked phones where applicable):")
        for _, row in final_df.head(10)[display_cols].iterrows():
            mp = dict(row)
            if 'suggested_phone' in mp and mp['suggested_phone']:
                mp['suggested_phone'] = _mask_phone(mp['suggested_phone'])
            logging.info(mp)
    except Exception:
        logging.debug("Failed to print top 10")

    return {
        "total": int(total),
        "verified": int(verified),
        "needs_update": int(needs_update),
        "needs_review": int(needs_review),
        "avg_confidence": round(float(avg_conf), 1),
        "output_file": output_csv,
        "elapsed_seconds": elapsed
    }
    try:
        run_id = push_results_to_supabase(final_df, {
            "total": total,
            "verified": verified,
            "needs_update": needs_update,
            "needs_review": needs_review,
            "avg_confidence": avg_confidence,
            "output_file": output_csv
        })
        logging.info(f"Pushed results to Supabase, run_id={run_id}")
    except Exception as e:
        logging.warning(f"Supabase push skipped/failed: {e}")
# ----------------------
# CLI entry
# ----------------------
if __name__ == "__main__":
    logging.info("Running solve_hack pipeline in CLI mode")
    summary = run_full_validation()
    logging.info(f"Summary: {summary}")
def push_results_to_supabase(final_df: pd.DataFrame, summary: dict) -> str:
    """
    Inserts one row into validation_runs and upserts all per-provider rows into validation_results.
    Returns run_id (str).
    """
    sb = get_supabase()
    run_id = str(uuid.uuid4())
    sb.table("validation_runs").insert({
        "run_id": run_id,
        "total": summary.get("total", 0),
        "verified": summary.get("verified", 0),
        "needs_update": summary.get("needs_update", 0),
        "needs_review": summary.get("needs_review", 0),
        "avg_confidence": summary.get("avg_confidence", 0.0),
        "output_file": summary.get("output_file", "")
    }).execute()
    cols = [
        "provider_id","name","phone","address","city","country","specialization",
        "npi_number","registration_number","status","confidence_score","suggested_phone",
        "npi_phone","registration_valid","google_phone","registry_source","source_agreement_count"
    ]
    payload_cols = [c for c in cols if c in final_df.columns]
    records = final_df[payload_cols].copy()
    records.insert(0, "run_id", run_id)

    # Convert NaN to None
    records = records.where(pd.notna(records), None)

    batch = []
    for rec in records.to_dict(orient="records"):
        batch.append(rec)
        if len(batch) >= 500:  # chunk
            sb.table("validation_results").upsert(batch).execute()
            batch = []
    if batch:
        sb.table("validation_results").upsert(batch).execute()

    return run_id
