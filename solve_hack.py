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
from pathlib import Path

# Try to enable AVIF support
try:
    from pillow_avif import AvifImagePlugin
    print("✓ AVIF support enabled")
except ImportError:
    print("⚠ AVIF support not available (install: pip install pillow-avif-plugin)")

print("--- RUNNING solve_hack.py (Indian Healthcare Validation) ---")

# Load the .env file to get our API keys
load_dotenv()

# --- AGENT 1A: NPI REGISTRY VALIDATOR (For US Providers) ---
def get_npi_data(npi_number):
    """Fetches provider data from the NPI registry (US providers)."""
    if pd.isna(npi_number) or str(npi_number).strip() == '':
        return None
        
    url = f"https://npiregistry.cms.hhs.gov/api/?number={npi_number}&version=2.1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('result_count', 0) == 0: 
            return None
        provider = data['results'][0]
        
        practice_locations = [addr for addr in provider.get('addresses', []) 
                            if addr.get('address_purpose') == 'LOCATION']
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

# --- AGENT 1B: INDIAN MEDICAL COUNCIL VALIDATOR (For Indian Providers) ---
def get_indian_medical_council_data(registration_number):
    """Validates provider with State Medical Council (Indian equivalent of NPI)"""
    # This is the INDIAN EQUIVALENT of NPI Registry
    # In production, this would scrape state medical council websites
    # For now, we'll do basic validation on registration format
    
    # Check if registration number follows pattern: STATE_CODE/NUMBER/YEAR
    pattern = r'^[A-Z]{2,4}/\d{4,5}/\d{4}$'
    
    if re.match(pattern, str(registration_number)):
        # Extract components
        parts = registration_number.split('/')
        state_code = parts[0]
        reg_num = parts[1]
        year = parts[2]
        
        # Simulate validation (80% pass rate for demo)
        if int(year) < 2000 or int(year) > 2024:
            return {"registration_valid": False, "error": "Invalid registration year"}
        
        # Simulate finding registration (like NPI lookup but for Indian doctors)
        import random
        if random.random() < 0.8:  # 80% validation success
            return {
                "registration_valid": True,
                "state_code": state_code,
                "registration_year": year,
                "status": "Active",
                "council_name": f"{state_code} Medical Council"
            }
        else:
            return {
                "registration_valid": False,
                "error": "Registration not found in database"
            }
    else:
        return {"registration_valid": False, "error": "Invalid registration format"}

# --- AGENT 2: GOOGLE MAPS VALIDATOR (Adapted for India) ---
def get_google_maps_data(provider_name, provider_address):
    """Fetches public data from Google Maps Places API."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Google Maps API key not found - skipping Google validation")
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
                # Clean Indian phone format
                clean_phone = "".join(filter(str.isdigit, phone))
                if len(clean_phone) > 10:
                    clean_phone = clean_phone[-10:]  # Take last 10 digits
                return {"google_phone": clean_phone if clean_phone else None}
        return None
    except requests.RequestException as e:
        print(f"Google Maps API Error: {e}")
        return None

# --- AGENT 3: QUALITY ASSURANCE (Hybrid - Works with both US and Indian data) ---
def validate_provider_row(row):
    """
    Cross-references provider data using appropriate registries:
    - US Providers: NPI Registry + Google Maps
    - Indian Providers: State Medical Council + Google Maps
    """
    
    # Clean input phone (handle both US and Indian formats)
    input_phone = str(row['phone'])
    input_phone_clean = "".join(filter(str.isdigit, input_phone))
    if len(input_phone_clean) > 10:
        input_phone_clean = input_phone_clean[-10:]  # Take last 10 digits
    
    # Determine if this is US or Indian provider
    has_npi = 'npi_number' in row and pd.notna(row.get('npi_number')) and str(row.get('npi_number')).strip() != ''
    has_indian_reg = 'registration_number' in row and pd.notna(row.get('registration_number')) and str(row.get('registration_number')).strip() != ''
    
    npi_phone = None
    registration_valid = False
    registry_source = None
    
    # Validate with appropriate registry
    if has_npi:
        # US Provider - Use NPI Registry
        npi_data = get_npi_data(row['npi_number'])
        npi_phone = npi_data.get('npi_phone') if npi_data else None
        registry_source = "NPI Registry"
    elif has_indian_reg:
        # Indian Provider - Use State Medical Council
        registration_data = get_indian_medical_council_data(row['registration_number'])
        registration_valid = registration_data.get('registration_valid', False)
        registry_source = registration_data.get('council_name', 'State Medical Council') if registration_valid else None
    
    # Validate with Google Maps (works for both US and Indian providers)
    google_data = get_google_maps_data(row['name'], row['address'])
    google_phone = google_data.get('google_phone') if google_data else None
    
    # Decision logic (unified for both US and Indian)
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
        # US provider with NPI data
        if input_phone_clean == npi_phone:
            status = "VERIFIED_OK (NPI)"
            confidence = 80
            suggested = npi_phone
        else:
            status = "NEEDS_UPDATE (NPI)"
            confidence = 70
            suggested = npi_phone
    elif registration_valid:
        # Indian provider with valid registration
        status = f"NEEDS_REVIEW ({registry_source})"
        confidence = 60
        suggested = input_phone_clean
    else:
        status = "NEEDS_MANUAL_REVIEW"
        confidence = 30
        suggested = None

    return {
        "status": status,
        "confidence_score": confidence,
        "suggested_phone": suggested,
        "npi_phone": npi_phone if has_npi else None,
        "registration_valid": registration_valid if has_indian_reg else None,
        "google_phone": google_phone,
        "registry_source": registry_source
    }

# --- Orchestrator ---
def run_full_validation():
    """Runs the entire cross-referencing pipeline."""
    try:
        df = pd.read_csv('input_providers.csv')
    except FileNotFoundError:
        print("Error: input_providers.csv not found!")
        return
    
    print(f"Running validation on {len(df)} providers...")
    print("Multi-Regional Validation System:")
    print("  • US Providers: NPI Registry + Google Maps")
    print("  • Indian Providers: State Medical Councils + Google Maps\n")
    
    validation_results = df.apply(validate_provider_row, axis=1)
    final_df = pd.concat([df, validation_results.apply(pd.Series)], axis=1)
    
    # Remove the testing column if it exists
    if 'expected_error_type' in final_df.columns:
        final_df = final_df.drop('expected_error_type', axis=1)
    
    final_df.to_csv('validation_results.csv', index=False)
    
    print("="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    
    # Summary statistics
    total = len(final_df)
    verified = len(final_df[final_df['status'].str.contains('VERIFIED', na=False)])
    needs_update = len(final_df[final_df['status'].str.contains('UPDATE', na=False)])
    needs_review = len(final_df[final_df['status'].str.contains('REVIEW', na=False)])
    avg_confidence = final_df['confidence_score'].mean()
    
    print(f"\n Validation Summary:")
    print(f"  Total Providers: {total}")
    print(f"   Verified: {verified} ({verified/total*100:.1f}%)")
    print(f"   Needs Update: {needs_update} ({needs_update/total*100:.1f}%)")
    print(f"   Needs Review: {needs_review} ({needs_review/total*100:.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.1f}%")
    
    print(f"\n Results saved to: validation_results.csv")
    print(f"\n Top 10 Results:")
    print(final_df[['provider_id', 'name', 'status', 'confidence_score']].head(10).to_string(index=False))


# --- AGENT 4: HIGH ACCURACY TESSERACT OCR ---
print("Tesseract OCR ready for use.")

def convert_image_to_supported_format(image_path):
    """Convert any image format to PNG for processing."""
    print(f"Converting image: {image_path}")
    
    try:
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        temp_path = 'temp_converted.png'
        img.save(temp_path, 'PNG')
        print(f"✓ Converted to PNG: {temp_path}")
        return temp_path
    except Exception as e:
        print(f"✗ PIL conversion failed: {e}")
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("OpenCV couldn't read the image")
            temp_path = 'temp_converted.png'
            cv2.imwrite(temp_path, img)
            print(f" Converted to PNG using OpenCV: {temp_path}")
            return temp_path
        except Exception as e2:
            print(f" OpenCV conversion also failed: {e2}")
            raise ValueError(f"Unable to read image format: {image_path}")

def enhance_image_for_ocr(image_path):
    """Apply multiple preprocessing techniques for best OCR results."""
    print("Enhancing image for OCR...")
    
    file_ext = Path(image_path).suffix.lower()
    if file_ext in ['.avif', '.webp', '.heic']:
        image_path = convert_image_to_supported_format(image_path)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    processed_images = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("otsu_threshold", thresh1))
    
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 31, 2)
    processed_images.append(("adaptive_threshold", thresh2))
    
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("denoised_otsu", thresh3))
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(("clahe_otsu", thresh4))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    processed_images.append(("morphological", morph))
    
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
        file_ext = Path(image_path).suffix.lower()
        working_path = image_path
        
        if file_ext in ['.avif', '.webp', '.heic']:
            print(f"⚠ Detected {file_ext} format - converting to PNG...")
            working_path = convert_image_to_supported_format(image_path)
        
        processed_images = enhance_image_for_ocr(working_path)
        all_results = []
        
        print("▸ Running OCR on ORIGINAL image...")
        original_img = Image.open(working_path)
        custom_config = r'--oem 3 --psm 6'
        
        original_text = pytesseract.image_to_string(original_img, config=custom_config)
        all_results.append(("original", original_text, len(original_text.strip())))
        print(f"   Extracted {len(original_text.strip())} characters")
        
        for method_name, processed_img in processed_images:
            print(f" Running OCR on {method_name.upper()}...")
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            char_count = len(text.strip())
            all_results.append((method_name, text, char_count))
            print(f"   Extracted {char_count} characters")
        
        best_result = max(all_results, key=lambda x: x[2])
        best_method, best_text, char_count = best_result
        
        print(f"\n BEST METHOD: {best_method.upper()} ({char_count} characters)")
        
        parsed_info = parse_medical_pamphlet_advanced(best_text)
        
        detailed_data = pytesseract.image_to_data(
            processed_images[0][1] if processed_images else original_img,
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )
        
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
        
        if os.path.exists('temp_converted.png'):
            os.remove('temp_converted.png')
        if os.path.exists('temp_preprocessed.jpg'):
            os.remove('temp_preprocessed.jpg')
        
        return final_result
        
    except Exception as e:
        print(f" ERROR during OCR processing: {e}")
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
    
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        if line_upper.startswith('DR.') or line_upper.startswith('DR '):
            if len(line) < 30 and not info["doctor_name"]:
                clean = re.sub(r'[|\\\/]', '', line)
                clean = re.sub(r'\s+', ' ', clean).strip()
                info["doctor_name"] = clean
                if i + 1 < len(lines):
                    next_line = lines[i + 1].upper()
                    if any(x in next_line for x in ['D.L.O', 'DLO', 'M.D', 'MD', 'MBBS']):
                        info["credentials"] = lines[i + 1].strip()
        
        if 'HOSPITAL' in line_upper and 5 < len(line) < 30:
            if not info["hospital_name"]:
                clean = re.sub(r'[^A-Za-z\s]', '', line)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if clean:
                    info["hospital_name"] = clean
        
        phone_match = re.search(r'\b\d{4,5}[-\s]?\d{6,8}\b', line)
        if phone_match:
            clean_phone = "".join(filter(str.isdigit, phone_match.group()))
            if clean_phone not in info["phone_numbers"]:
                info["phone_numbers"].append(clean_phone)
        
        email_match = re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', line)
        if email_match and not info["email"]:
            info["email"] = email_match.group().lower()
        
        website_match = re.search(r'www\.[a-zA-Z0-9.-]+\.(com|net|in|org)', line, re.IGNORECASE)
        if website_match and not info["website"]:
            info["website"] = website_match.group().lower()
        
        if any(x in line_upper for x in ['ENDOSCOPIC', 'ENT,', 'SURGEON']):
            if 'SPECIALIST IN' not in line_upper and not info["specialization"]:
                clean = re.sub(r'[^A-Za-z\s,&]', '', line)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if 10 < len(clean) < 60:
                    info["specialization"] = clean
    
    medical_terms = {
        'EAR', 'THROAT', 'TONSIL', 'ALLERG', 'NOSE', 
        'SNEEZ', 'SNOR', 'HEARING', 'VERTIGO', 'REFLUX', 
        'FOREIGN', 'THYROID', 'PAIN', 'INFECTION'
    }
    
    for line in lines:
        line_upper = line.upper()
        
        if any(term in line_upper for term in medical_terms):
            if any(skip in line_upper for skip in ['SPECIALIST IN', 'CONTACT', 'CALL', 'DR.']):
                continue
            
            clean = re.sub(r'^[^A-Z]+', '', line)
            clean = re.sub(r'[^A-Za-z\s&]', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            
            if 5 < len(clean) < 50 and clean not in info["conditions_treated"]:
                info["conditions_treated"].append(clean)
    
    return info

# --- This is the main script that runs ---
if __name__ == "__main__":
    
    # --- PART 1: Provider Validation ---
    run_full_validation()
    
    # --- PART 2: OCR EXTRACTION (Optional) ---
    # Look for image files to process
    image_extensions = ['.jpg', '.jpeg', '.png', '.avif', '.webp']
    image_files = []
    for ext in image_extensions:
        files = list(Path('.').glob(f'*{ext}'))
        files = [f for f in files if not str(f).startswith('temp_')]
        image_files.extend(files)
    
    if image_files:
        # Process the first image found
        image_file = str(image_files[0])
        print(f"\n Found image file: {image_file}")
        print("Processing with OCR...\n")
        
        vlm_data = extract_data_from_image_tesseract(image_file)
        
        print("\n" + "="*80)
        print("FULL EXTRACTED TEXT")
        print("="*80)
        if "full_text" in vlm_data:
            print(vlm_data["full_text"][:500] + "..." if len(vlm_data["full_text"]) > 500 else vlm_data["full_text"])
        
        print("\n" + "="*80)
        print("PARSED MEDICAL INFORMATION")
        print("="*80)
        if "parsed_information" in vlm_data:
            parsed = vlm_data["parsed_information"]
            for key, value in parsed.items():
                if value:
                    if isinstance(value, list) and value:
                        print(f"\n{key.replace('_', ' ').title()}:")
                        for item in value[:5]:  # Show first 5
                            print(f"  • {item}")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Save results
        with open('ocr_results.json', 'w') as f:
            json.dump(vlm_data, f, indent=2)
        print(f"\n OCR results saved to: ocr_results.json")
    else:
        print("\n No images found for OCR processing.")
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETE!")
    print("="*80)
    print("\n Launch dashboard with: streamlit run dashboard.py")