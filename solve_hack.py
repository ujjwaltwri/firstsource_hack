import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json

# --- VLM Imports ---
import google.generativeai as genai
from PIL import Image

# Load the .env file to get our API keys
load_dotenv()

# --- VLM CLIENT (GEMINI) ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # We will try the 'gemini-pro-vision' model again
    # The 404 error is an account/API key issue, not a code issue.
    gemini_model = genai.GenerativeModel('gemini-pro-vision')
    print("Gemini client initialized successfully.")
except Exception as e:
    print(f"Warning: Gemini client could not be initialized. VLM will not work. Error: {e}")
    gemini_model = None

# --- AGENT 1: NPI DATA VALIDATOR ---
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

# --- AGENT 2: GOOGLE MAPS VALIDATOR ---
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

# --- AGENT 3: QUALITY ASSURANCE (The "Brain") ---
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

# --- Orchestrator ---
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

# --- Information Enrichment Agent (VLM) ---
def extract_data_from_image(image_path):
    if not gemini_model:
        return {"error": "Gemini client is not initialized. Check API key."}
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return {"error": "Could not read image file."}
    except Exception as e:
        print(f"Error opening image: {e}")
        return {"error": str(e)}

    prompt = """
    You are an AI assistant for healthcare data management.
    Analyze this image, which is a scanned document for a provider.
    Extract the following information and return it ONLY as a valid JSON object:
    - "name" (string)
    - "phone" (string, just the numbers)
    - "address" (string, full address)
    If data is missing, set the value to null.
    Do not add any text before or after the JSON.
    """
    
    print(f"\n--- Starting Gemini VLM extraction for {image_path} ---")
    
    try:
        response = gemini_model.generate_content([prompt, img])
        raw_text = response.text
        print("Gemini Model Raw Output:", raw_text)
        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1
        json_string = raw_text[json_start:json_end]
        json_data = json.loads(json_string)
        return json_data
    except Exception as e:
        print(f"Error with Gemini API call: {e}")
        return {"error": str(e)}

# --- This allows us to run this file directly OR import from it ---
if __name__ == "__main__":
    run_full_validation()
    
    print("\n--- Testing VLM Extraction ---")
    vlm_data = extract_data_from_image('pamplet 4.jpeg')
    print("\nExtracted VLM Data (as JSON):")
    print(vlm_data)