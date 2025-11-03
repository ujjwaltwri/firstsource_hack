import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json
from PIL import Image

# --- PaddleOCR Imports ---
# This is our new, 100% free VLM
from paddleocr import PaddleOCR

print("--- RUNNING solve_hack.py (Correct VLM / Correct NPI) ---")

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


# --- AGENT 4: FREE VLM (CORRECT MODEL: PaddleOCR) ---
print("Loading free PaddleOCR VLM model...")
# This initializes the OCR model. lang='en' for English.
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
print("Free PaddleOCR VLM model loaded successfully.")

def extract_data_from_image_free(image_path):
    """Uses a 100% free PaddleOCR model to READ the image."""
        
    print(f"\n--- Starting FREE VLM (PaddleOCR) extraction for {image_path} ---")
    
    try:
        # --- FIX #2: We are using PaddleOCR. It's built for this. ---
        # It will return a list of all text blocks it finds.
        result = ocr.ocr(image_path)
        
        # We'll just extract the text parts for a clean list
        extracted_text_list = []
        if result and result[0]:
            for line in result[0]:
                extracted_text_list.append(line[1][0]) # line[1][0] is the text

        final_result = {
            "model_used": "PaddleOCR",
            "extracted_lines": extracted_text_list
        }
        
        print("Free VLM (PaddleOCR) extraction complete.")
        return final_result
        
    except Exception as e:
        print(f"Error during VLM processing: {e}")
        return {"error": str(e)}

# --- This is the main script that runs ---
if __name__ == "__main__":
    
    # --- PART 1: NPI + Google Maps Validation ---
    run_full_validation()
    
    # --- PART 2: VLM EXTRACTION ---
    vlm_data = extract_data_from_image_free('pamplet 4.jpeg')
    print("\nExtracted VLM Data (100% Free OCR):")
    print(vlm_data)