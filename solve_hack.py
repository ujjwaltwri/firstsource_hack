import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json
from PIL import Image
import cv2
import numpy as np
import pytesseract
import re

print("--- RUNNING solve_hack.py (HIGH ACCURACY Tesseract OCR) ---")

# Load the .env file to get our API keys
load_dotenv()

# --- AGENT 1: NPI DATA VALIDATOR (Working) ---
def get_npi_data(npi_number):
    """Fetches provider data from the NPI registry."""
    url = f"https://npiregistry.cms.hhs.gov/api/?number={npi_number}&version=2.1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('result_count', 0) == 0: return None
        provider = data['results'][0]
        
        practice_locations = [addr for addr in provider.get('addresses', []) if addr.get('address_purpose') == 'LOCATION']
        clean_phone, npi_name = None, None
        
        for loc in practice_locations:
            raw_phone = loc.get('telephone_number', '')
            if raw_phone:
                clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                break
        
        if not clean_phone:
            for addr in provider.get('addresses', []):
                if addr.get('address_purpose') == 'MAILING':
                    raw_phone = addr.get('telephone_number', '')
                    if raw_phone:
                        clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                        break
        
        basic_info = provider.get('basic', {})
        if basic_info.get('first_name'):
            npi_name = f"{basic_info.get('first_name', '')} {basic_info.get('last_name', '')}"
        elif basic_info.get('organization_name'):
            npi_name = basic_info.get('organization_name')
        
        return {"npi_phone": clean_phone, "npi_name": npi_name}
    except requests.RequestException as e:
        print(f"NPI API Error: {e}")
        return None

# --- AGENT 2: GOOGLE MAPS VALIDATOR (Working) ---
def get_google_maps_data(provider_name, provider_address):
    """Fetches public data from Google Maps Places API."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Google Maps API key not found!")
        return None
        
    search_query = f"{provider_name} {provider_address}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {'query': search_query, 'key': api_key}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            place_id = data['results'][0]['place_id']
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {'place_id': place_id, 'fields': 'name,formatted_phone_number', 'key': api_key}
            details_response = requests.get(details_url, params=details_params, timeout=5)
            details_data = details_response.json()
            
            if details_data.get('status') == 'OK' and details_data.get('result'):
                phone = details_data['result'].get('formatted_phone_number', '')
                return {"google_phone": "".join(filter(str.isdigit, phone))[:10]}
        return None
    except requests.RequestException as e:
        print(f"Google Maps API Error: {e}")
        return None

# --- AGENT 3: QUALITY ASSURANCE (Working) ---
def validate_provider_row(row):
    """Cross-references NPI and Google Maps to find the truth."""
    input_phone = "".join(filter(str.isdigit, str(row['phone'])))[:10]
    npi_data = get_npi_data(row['npi_number'])
    google_data = get_google_maps_data(row['name'], row['address'])
    
    npi_phone = npi_data.get('npi_phone') if npi_data else None
    google_phone = google_data.get('google_phone') if google_data else None
    
    if input_phone == google_phone:
        status, confidence, suggested = "VERIFIED_OK (Google)", 100, google_phone
    elif input_phone == npi_phone:
        status, confidence, suggested = "VERIFIED_OK (NPI)", 80, npi_phone
    elif google_phone:
        status, confidence, suggested = "UPDATED (Google)", 95, google_phone
    elif npi_phone:
        status, confidence, suggested = "NEEDS_REVIEW (NPI)", 40, npi_phone
    else:
        status, confidence, suggested = "ERROR_ALL_SOURCES", 0, None

    return {
        "status": status,
        "confidence_score": confidence,
        "suggested_phone": suggested,
        "npi_phone": npi_phone,
        "google_phone": google_phone
    }

# --- Orchestrator (Working) ---
def run_full_validation():
    """Runs the entire cross-referencing pipeline."""
    try:
        df = pd.read_csv('input_providers.csv')
    except FileNotFoundError:
        print("Error: input_providers.csv not found!")
        return
        
    print("Running FULL validation (NPI + Google Maps)...")
    validation_results = df.apply(validate_provider_row, axis=1)
    final_df = pd.concat([df, validation_results.apply(pd.Series)], axis=1)
    
    final_df.to_csv('validation_results.csv', index=False)
    print("Validation complete! Results saved to 'validation_results.csv'")
    print(final_df)


# --- AGENT 4: HIGH ACCURACY TESSERACT OCR ---
print("Tesseract OCR ready for use.")

def enhance_image_for_ocr(image_path):
    """Apply multiple preprocessing techniques for best OCR results."""
    print("Enhancing image for OCR...")
    
    # Read image
    img = cv2.imread(image_path)
    
    # Multiple preprocessing pipelines
    processed_images = []
    
    # Method 1: Standard grayscale + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("otsu_threshold", thresh1))
    
    # Method 2: Adaptive threshold
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 31, 2)
    processed_images.append(("adaptive_threshold", thresh2))
    
    # Method 3: Denoise + threshold
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("denoised_otsu", thresh3))
    
    # Method 4: Contrast enhancement + threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("clahe_otsu", thresh4))
    
    # Method 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    processed_images.append(("morphological", morph))
    
    # Method 6: Bilateral filter + threshold
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh5 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("bilateral_otsu", thresh5))
    
    return processed_images

def extract_data_from_image_tesseract(image_path):
    """Uses Tesseract OCR with multiple preprocessing methods for highest accuracy."""
    
    print(f"\n{'='*80}")
    print(f"STARTING HIGH-ACCURACY TESSERACT OCR: {image_path}")
    print(f"{'='*80}\n")
    
    try:
        # Get multiple preprocessed versions
        processed_images = enhance_image_for_ocr(image_path)
        
        all_results = []
        
        # Try OCR on original image first
        print("▸ Running OCR on ORIGINAL image...")
        original_img = Image.open(image_path)
        
        # Tesseract configuration for best accuracy
        custom_config = r'--oem 3 --psm 6'  # OEM 3 = Default, PSM 6 = Uniform block of text
        
        original_text = pytesseract.image_to_string(original_img, config=custom_config)
        all_results.append(("original", original_text, len(original_text.strip())))
        print(f"  ✓ Extracted {len(original_text.strip())} characters")
        
        # Try OCR on each preprocessed version
        for method_name, processed_img in processed_images:
            print(f"▸ Running OCR on {method_name.upper()}...")
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            char_count = len(text.strip())
            all_results.append((method_name, text, char_count))
            print(f"  ✓ Extracted {char_count} characters")
        
        # Select the result with the most extracted text
        best_result = max(all_results, key=lambda x: x[2])
        best_method, best_text, char_count = best_result
        
        print(f"\n✓ BEST METHOD: {best_method.upper()} ({char_count} characters)")
        
        # Parse the extracted text
        parsed_info = parse_medical_pamphlet_advanced(best_text)
        
        # Get detailed text with confidence (using image_to_data)
        detailed_data = pytesseract.image_to_data(
            processed_images[0][1] if processed_images else original_img,
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )
        
        # Extract lines with confidence scores
        lines_with_confidence = []
        for i, text in enumerate(detailed_data['text']):
            if text.strip():
                lines_with_confidence.append({
                    "text": text.strip(),
                    "confidence": detailed_data['conf'][i]
                })
        
        final_result = {
            "model_used": "Tesseract OCR",
            "best_preprocessing_method": best_method,
            "total_characters_extracted": char_count,
            "full_text": best_text,
            "lines_with_confidence": lines_with_confidence,
            "parsed_information": parsed_info,
            "all_methods_tested": [m[0] for m in all_results]
        }
        
        return final_result
        
    except Exception as e:
        print(f"✗ ERROR during OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def parse_medical_pamphlet_advanced(text):
    """Simplified parsing with better logic."""
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
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # First pass: find key information with strict rules
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        # Doctor name: must start with DR and be short
        if line_upper.startswith('DR.') or line_upper.startswith('DR '):
            if len(line) < 30 and not info["doctor_name"]:
                clean = re.sub(r'[|\\\/]', '', line)
                clean = re.sub(r'\s+', ' ', clean).strip()
                info["doctor_name"] = clean
                # Next line might be credentials
                if i + 1 < len(lines):
                    next_line = lines[i + 1].upper()
                    if any(x in next_line for x in ['D.L.O', 'DLO', 'M.D', 'MD', 'MBBS']):
                        info["credentials"] = lines[i + 1].strip()
        
        # Hospital: must contain HOSPITAL and be reasonably short
        if 'HOSPITAL' in line_upper and 5 < len(line) < 30:
            if not info["hospital_name"]:
                clean = re.sub(r'[^A-Za-z\s]', '', line)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if clean:
                    info["hospital_name"] = clean
        
        # Phone: strict pattern matching
        phone_match = re.search(r'\b\d{5}[-\s]?\d{6}\b', line)
        if phone_match:
            clean_phone = "".join(filter(str.isdigit, phone_match.group()))
            if clean_phone not in info["phone_numbers"]:
                info["phone_numbers"].append(clean_phone)
        
        # Email: strict email pattern
        email_match = re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', line)
        if email_match and not info["email"]:
            info["email"] = email_match.group().lower()
        
        # Website: strict www pattern
        website_match = re.search(r'www\.[a-zA-Z0-9.-]+\.(com|net|in|org)', line, re.IGNORECASE)
        if website_match and not info["website"]:
            info["website"] = website_match.group().lower()
        
        # Specialization: only lines with ENT/SURGEON/ENDOSCOPIC
        if any(x in line_upper for x in ['ENDOSCOPIC', 'ENT,', 'SURGEON']):
            if 'SPECIALIST IN' not in line_upper and not info["specialization"]:
                clean = re.sub(r'[^A-Za-z\s,&]', '', line)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if 10 < len(clean) < 60:
                    info["specialization"] = clean
    
    # Second pass: extract conditions (only lines with medical terms)
    medical_terms = {
        'EAR', 'THROAT', 'TONSIL', 'ALLERG', 'NOSE', 
        'SNEEZ', 'SNOR', 'HEARING', 'VERTIGO', 'REFLUX', 
        'FOREIGN', 'THYROID', 'PAIN', 'INFECTION'
    }
    
    for line in lines:
        line_upper = line.upper()
        
        # Must contain a medical term
        if any(term in line_upper for term in medical_terms):
            # Skip if it's metadata
            if any(skip in line_upper for skip in ['SPECIALIST IN', 'CONTACT', 'CALL', 'DR.']):
                continue
            
            # Clean the line
            clean = re.sub(r'^[^A-Z]+', '', line)  # Remove leading junk
            clean = re.sub(r'[^A-Za-z\s&]', '', clean)  # Keep only letters and &
            clean = re.sub(r'\s+', ' ', clean).strip()  # Normalize spaces
            
            # Add if it's reasonable
            if 5 < len(clean) < 50 and clean not in info["conditions_treated"]:
                info["conditions_treated"].append(clean)
    
    return info

# --- This is the main script that runs ---
if __name__ == "__main__":
    
    # --- PART 1: NPI + Google Maps Validation ---
    run_full_validation()
    
    # --- PART 2: HIGH-ACCURACY TESSERACT OCR EXTRACTION ---
    vlm_data = extract_data_from_image_tesseract('pamplet 4.jpeg')
    
    print("\n" + "="*80)
    print("FULL EXTRACTED TEXT")
    print("="*80)
    if "full_text" in vlm_data:
        print(vlm_data["full_text"])
    
    print("\n" + "="*80)
    print("PARSED MEDICAL INFORMATION")
    print("="*80)
    if "parsed_information" in vlm_data:
        parsed = vlm_data["parsed_information"]
        for key, value in parsed.items():
            if value:
                if isinstance(value, list) and value:
                    print(f"\n{key.replace('_', ' ').title()}:")
                    for item in value:
                        print(f"  • {item}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Save detailed results
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)
    with open('ocr_results.json', 'w') as f:
        json.dump(vlm_data, f, indent=2)
    print("✓ Detailed OCR results saved to 'ocr_results.json'")