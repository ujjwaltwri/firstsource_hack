import requests
import pandas as pd

# --- Data Validation Agent ---
def get_npi_data(npi_number):
    """
    Fetches provider data from the NPI registry.
    This is part of the Data Validation Agent's job.
    """
    # Define the API endpoint
    url = f"https://npiregistry.cms.hhs.gov/api/?number={npi_number}&version=2.1"
    
    try:
        # Make the API call
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad responses (like 404, 500)
        data = response.json()
        
        # Check if the API returned any results
        if data.get('result_count', 0) > 0:
            provider = data['results'][0]
            
            # Navigate the JSON to find the practice address (we'll assume the first one)
            practice_address = None
            for addr in provider.get('addresses', []):
                if addr.get('address_purpose') == 'LOCATION':
                    practice_address = addr
                    break
            
            # If no 'LOCATION' address, just grab the first one
            if not practice_address and provider.get('addresses'):
                practice_address = provider['addresses'][0]

            if practice_address:
                # Clean up the phone number (remove dashes, spaces)
                raw_phone = practice_address.get('telephone_number', '')
                clean_phone = "".join(filter(str.isdigit, raw_phone))
                
                # Format the address for comparison
                full_address = f"{practice_address.get('address_1', '')}, {practice_address.get('city', '')}, {practice_address.get('state', '')}"
                
                return {
                    "npi_phone": clean_phone[:10], # Get standard 10-digit phone
                    "npi_address": full_address,
                    "npi_name": f"{provider['basic']['first_name']} {provider['basic']['last_name']}"
                }

        # If no results or no address
        return None

    except requests.RequestException as e:
        print(f"Error fetching NPI data for {npi_number}: {e}")
        return None

# --- Quality Assurance Agent ---
def validate_provider_row(row):
    """
    Compares input data to NPI data and generates a confidence score.
    This is the Quality Assurance Agent's job.
    """
    # Clean the input phone number for a fair comparison
    input_phone = "".join(filter(str.isdigit, str(row['phone'])))[:10]
    
    # Call our other agent to get the NPI data
    npi_data = get_npi_data(row['npi_number'])
    
    # Case 1: We couldn't find the provider in the NPI registry
    if not npi_data:
        return {
            "status": "ERROR_NPI_FETCH",
            "confidence_score": 0,
            "suggested_phone": None,
            "npi_phone": None
        }

    # Case 2: The input phone MATCHES the NPI phone
    if input_phone == npi_data['npi_phone']:
        status = "VERIFIED_OK"
        confidence = 100
        suggested_phone = input_phone
    
    # Case 3: The phones DO NOT MATCH
    else:
        status = "NEEDS_MANUAL_REVIEW"
        confidence = 40  # Low confidence, as NPI data can also be outdated
        suggested_phone = npi_data['npi_phone']
        
    return {
        "status": status,
        "confidence_score": confidence,
        "suggested_phone": suggested_phone,
        "npi_phone": npi_data['npi_phone'] # Include what the NPI registry said
    }

# --- Orchestrator ---
def run_validation():
    """
    Runs the entire validation process on the input CSV.
    This function acts as the main orchestrator.
    """
    try:
        df = pd.read_csv('input_providers.csv')
    except FileNotFoundError:
        print("Error: input_providers.csv not found!")
        return
    
    print("Running validation on input data...")
    
    # This is the magic!
    # df.apply() runs our 'validate_provider_row' agent on every single row
    validation_results = df.apply(validate_provider_row, axis=1)
    
    # Combine the original data (df) with the new results
    final_df = pd.concat([df, validation_results.apply(pd.Series)], axis=1)
    
    # Save the results to a new CSV file
    final_df.to_csv('validation_results.csv', index=False)
    
    print("Validation complete! Results saved to 'validation_results.csv'")
    print(final_df)
    return final_df