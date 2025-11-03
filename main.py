from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import Response  # <-- The important import for the fix
import pandas as pd

app = FastAPI()

# --- THIS IS THE FIXED FUNCTION ---
@app.get("/api/validation-results", response_class=Response)
def get_validation_results():
    """
    API endpoint to read and serve the validation results.
    This is part of the Directory Management Agent's job.
    """
    try:
        # Read the results from the CSV we made in Phase 2
        df = pd.read_csv('validation_results.csv')
        
        # Use pandas' built-in .to_json() which correctly handles NaN,
        # converting them to 'null' for JSON.
        json_output = df.to_json(orient="records")
        
        # Return it as a JSON response
        return Response(content=json_output, media_type="application/json")
        
    except FileNotFoundError:
        # If the run.py script hasn't been run yet
        return Response(content='{"error": "Results file not found. Run \'run.py\' script first."}', 
                        media_type="application/json", status_code=404)
    except Exception as e:
        return Response(content=f'{{"error": "{str(e)}"}}', 
                        media_type="application/json", status_code=500)

@app.get("/api/generate-email/{provider_id}")
def generate_provider_email(provider_id: int):
    """
    Generates a communication email for a specific flagged provider.
    This is also part of the Directory Management Agent's job.
    """
    try:
        df = pd.read_csv('validation_results.csv')
        
        # Find the specific provider row by its ID
        provider_row = df[df['provider_id'] == provider_id]
        
        if provider_row.empty:
            return {"error": "Provider ID not found."}
            
        # Get the first (and only) row as a dictionary
        provider = provider_row.iloc[0].to_dict()
        
        # Only generate a detailed email if they need review
        if provider['status'] == "NEEDS_MANUAL_REVIEW":
            email_body = f"""
            Subject: Action Required: Please verify your provider directory information
            
            Dear {provider['name']},
            
            Our automated systems detected a potential discrepancy in your contact information during a routine check.
            
            - Our Current Record (Phone): {provider['phone']}
            - Publicly Listed Record (NPI): {provider['suggested_phone']}
            
            To ensure our members can reach you, please contact us to confirm your correct details.
            
            Thank you,
            Healthcare Payer Team
            """
            return {"provider_id": provider_id, "email_generated": email_body}
        
        # If their status is VERIFIED_OK
        return {"provider_id": provider_id, "email_generated": "No action needed, provider is verified."}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """
    Serves the main HTML dashboard (index.html).
    """
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found!</h1>"