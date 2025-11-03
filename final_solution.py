import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json
from PIL import Image

# --- NEW: Hugging Face Imports (THE FIX) ---
# We ONLY need pipeline. The other imports were wrong.
from transformers import pipeline

# --- AGENT 1: NPI DATA VALIDATOR ---
# This is the "smart loop" logic that *will* run now
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
        
        if not clean_phone:
            print(f"NPI {npi_number}: No phone number found.")
            return None

        basic_info = provider.get('basic', {})
        if basic_info.get('first_name'):
            npi_name = f"{basic_info.get('first_name', '')} {basic_info.get('last_name', '')}"
        elif basic_info.get('organization_name'):
            npi_name = basic_info.get('organization_name')
        
        return {"npi_phone": clean_phone, "npi_name": npi_name}
    except requests.RequestException as e:
        print(f"NPI API Error: {e}")
        return None

# --- AGENT 2: QUALITY ASSURANCE (NPI-Only) ---
def validate_provider_row(row):
    """Compares input data to NPI data."""
    input_phone = "".join(filter(str.isdigit, str(row['phone'])))[:10]
    npi_data = get_npi_data(row['npi_number'])
    
    if not npi_data:
        return {"status": "ERROR_NPI_FETCH", "confidence_score": 0, "suggested_phone": None, "npi_phone": None}
    if input_phone == npi_data['npi_phone']:
        status, confidence, suggested = "VERIFIED_OK (NPI)", 100, input_phone
    else:
        status, confidence, suggested = "NEEDS_REVIEW (NPI)", 40, npi_data['npi_phone']
        
    return {"status": status, "confidence_score": confidence, "suggested_phone": suggested, "npi_phone": npi_data['npi_phone']}

# --- Orchestrator ---
def run_npi_validation():
    """Runs the entire cross-referencing pipeline."""
    try:
        df = pd.read_csv('input_providers.csv')
    except FileNotFoundError:
        print("Error: input_providers.csv not found!")
        return
        
    print("Running NPI-Only validation...")
    validation_results = df.apply(validate_provider_row, axis=1)
    final_df = pd.concat([df, validation_results.apply(pd.Series)], axis=1)
    
    final_df.to_csv('validation_results.csv', index=False)
    print("NPI Validation complete! Results saved to 'validation_results.csv'")
    print(final_df)

# --- AGENT 3: FREE VLM (Hugging Face) ---
# This will download the model the first time you run it (about 1.7GB)
print("Loading free VLM model... (This may take a minute the first time)")
# We use a Visual Question Answering (VQA) model
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
print("Free VLM model loaded successfully.")

def extract_data_from_image_free(image_path):
    """Uses a 100% free Hugging Face model to get info."""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return {"error": f"Image not found at {image_path}"}
        
    print(f"\n--- Starting FREE VLM extraction for {image_path} ---")
    
    try:
        # We have to ask it questions one by one
        q_name = "What is the doctor's name?"
        q_phone = "What is the phone number?"
        
        # Run the model for each question
        name_result = vqa_pipeline(img, question=q_name, top_k=1)
        phone_result = vqa_pipeline(img, question=q_phone, top_k=1)
        
        # Extract the answers
        extracted_name = name_result[0]['answer']
        extracted_phone = phone_result[0]['answer']

        # This free model is not as good as Gemini, so we print the raw answers
        result = {
            "extracted_name": extracted_name,
            "extracted_phone": extracted_phone
        }
        
        print("Free VLM extraction complete.")
        return result
        
    except Exception as e:
        print(f"Error during VLM processing: {e}")
        return {"error": str(e)}

# --- This is the main script that runs ---
if __name__ == "__main__":
    
    # --- PART 1: NPI VALIDATION ---
    run_npi_validation()
    
    # --- PART 2: VLM EXTRACTION ---
    vlm_data = extract_data_from_image_free('pamplet 4.jpeg')
    print("\nExtracted VLM Data (100% Free):")
    print(vlm_data)