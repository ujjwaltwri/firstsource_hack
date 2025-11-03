import requests
import pandas as pd
import os
import json

print("--- RUNNING FINAL_TEST.PY (THE 100% CORRECT SCRIPT) ---")

# --- AGENT 1: NPI DATA VALIDATOR (TRULY FIXED LOGIC) ---
def get_npi_data(npi_number):
    """Fetches provider data from the NPI registry."""
    url = f"https://npiregistry.cms.hhs.gov/api/?number={npi_number}&version=2.1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('result_count', 0) == 0: return None
        provider = data['results'][0]
        
        # --- FIX #1: We get ALL phones from ALL locations ---
        all_phones = []
        for addr in provider.get('addresses', []):
            # We check both LOCATION and MAILING addresses
            if addr.get('address_purpose') in ['LOCATION', 'MAILING']:
                raw_phone = addr.get('telephone_number', '')
                if raw_phone:
                    clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                    if clean_phone not in all_phones:
                        all_phones.append(clean_phone)
        
        if not all_phones:
            print(f"NPI {npi_number}: No phone number found.")
            return None

        # Get the provider's name
        basic_info = provider.get('basic', {})
        npi_name = ""
        if basic_info.get('first_name'):
            npi_name = f"{basic_info.get('first_name', '')} {basic_info.get('last_name', '')}"
        elif basic_info.get('organization_name'):
            npi_name = basic_info.get('organization_name')
        
        return {"npi_phones": all_phones, "npi_name": npi_name}
    except requests.RequestException as e:
        print(f"NPI API Error: {e}")
        return None

# --- AGENT 2: QUALITY ASSURANCE (TRULY FIXED LOGIC) ---
def validate_provider_row(row):
    """Compares input data to NPI data using the new smart logic."""
    input_phone = "".join(filter(str.isdigit, str(row['phone'])))[:10]
    npi_data = get_npi_data(row['npi_number'])
    
    if not npi_data:
        return {"status": "ERROR_NPI_FETCH", "confidence_score": 0, "suggested_phone": None, "npi_phones": None}

    # --- THIS IS THE FIX ---
    # We check if our input_phone is IN the list of *all* NPI phones
    # This will find the '703' number for Leesburg and mark it VERIFIED
    if input_phone in npi_data['npi_phones']:
        status = "VERIFIED_OK (NPI)"
        confidence = 100
        suggested = input_phone
    else:
        # If it doesn't match, just suggest the *first* phone found
        status = "NEEDS_REVIEW (NPI)"
        confidence = 40
        suggested = npi_data['npi_phones'][0] 
        
    return {
        "status": status, 
        "confidence_score": confidence, 
        "suggested_phone": suggested, 
        "npi_phones": ", ".join(npi_data['npi_phones']) # Store all for debugging
    }

# --- Orchestrator ---
def run_npi_validation():
    """Runs the entire NPI validation pipeline."""
    try:
        # We need the corrected CSV
        df = pd.read_csv('input_providers.csv')
    except FileNotFoundError:
        print("Error: input_providers.csv not found!")
        return
        
    print("Running NPI Validation (with correct logic)...")
    validation_results = df.apply(validate_provider_row, axis=1)
    final_df = pd.concat([df, validation_results.apply(pd.Series)], axis=1)
    
    final_df.to_csv('validation_results.csv', index=False)
    print("NPI Validation complete! Results saved to 'validation_results.csv'")
    print(final_df)

# --- This is the main script that runs ---
if __name__ == "__main__":
    
    # --- We will ONLY run the NPI validation ---
    run_npi_validation()