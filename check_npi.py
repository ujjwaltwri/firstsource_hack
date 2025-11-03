import requests
import pandas as pd

def get_npi_data(npi_number):
    """Fetches provider data from the NPI registry."""
    url = f"https://npiregistry.cms.hhs.gov/api/?number={npi_number}&version=2.1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get('result_count', 0) == 0: return None
        provider = data['results'][0]
        
        # --- THIS IS THE "SMART LOGIC" THAT WILL FINALLY RUN ---
        practice_locations = [addr for addr in provider.get('addresses', []) if addr.get('address_purpose') == 'LOCATION']
        clean_phone, full_address = None, None
        
        for loc in practice_locations:
            raw_phone = loc.get('telephone_number', '')
            if raw_phone:
                clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                full_address = f"{loc.get('address_1', '')}, {loc.get('city', '')}, {loc.get('state', '')}"
                break
        
        if not clean_phone:
            for addr in provider.get('addresses', []):
                if addr.get('address_purpose') == 'MAILING':
                    raw_phone = addr.get('telephone_number', '')
                    if raw_phone:
                        clean_phone = "".join(filter(str.isdigit, raw_phone))[:10]
                        full_address = f"{addr.get('address_1', '')}, {addr.get('city', '')}, {addr.get('state', '')}"
                        break
        
        if not clean_phone:
            print(f"NPI {npi_number}: No phone number found.")
            return None
        # --- END OF SMART LOGIC ---

        basic_info = provider.get('basic', {})
        npi_name = ""
        if basic_info.get('first_name'):
            npi_name = f"{basic_info.get('first_name', '')} {basic_info.get('last_name', '')}"
        elif basic_info.get('organization_name'):
            npi_name = basic_info.get('organization_name')
        else: npi_name = "Name Not Found"
        
        return {"npi_phone": clean_phone, "npi_address": full_address, "npi_name": npi_name.strip()}
    except requests.RequestException as e:
        print(f"Error fetching NPI data for {npi_number}: {e}")
        return None

def validate_provider_row(row):
    """Compares input data to NPI data."""
    input_phone = "".join(filter(str.isdigit, str(row['phone'])))[:10]
    npi_data = get_npi_data(row['npi_number'])
    
    if not npi_data:
        return {"status": "ERROR_NPI_FETCH", "confidence_score": 0, "suggested_phone": None, "npi_phone": None}
    if input_phone == npi_data['npi_phone']:
        status, confidence, suggested = "VERIFIED_OK", 100, input_phone
    else:
        status, confidence, suggested = "NEEDS_MANUAL_REVIEW", 40, npi_data['npi_phone']
    return {"status": status, "confidence_score": confidence, "suggested_phone": suggested, "npi_phone": npi_data['npi_phone']}

def run_validation():
    """Runs the entire validation process."""
    try:
        df = pd.read_csv('input_providers.csv')
    except FileNotFoundError:
        print("Error: input_providers.csv not found!")
        return
    print("Running validation on input data...")
    validation_results = df.apply(validate_provider_row, axis=1)
    final_df = pd.concat([df, validation_results.apply(pd.Series)], axis=1)
    final_df.to_csv('validation_results.csv', index=False)
    print("Validation complete! Results saved to 'validation_results.csv'")
    print(final_df)

if __name__ == "__main__":
    print("--- Running check_npi.py ---")
    run_validation()
    print("--- NPI Check Complete ---")
