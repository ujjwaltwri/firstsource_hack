from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
import pandas as pd
import json

# Import our new, working functions from solve_hack.py
from solve_hack import run_full_validation, extract_data_from_image

app = FastAPI()

@app.post("/api/run-validation")
def run_validation_endpoint():
    """Runs the validation process and returns the results."""
    try:
        run_full_validation() # This re-creates validation_results.csv
        return {"message": "Validation complete!"}
    except Exception as e:
        return Response(content=f'{{"error": "{str(e)}"}}', media_type="application/json", status_code=500)

@app.get("/api/validation-results")
def get_validation_results():
    """API endpoint to read and serve the validation results."""
    try:
        df = pd.read_csv('validation_results.csv')
        json_output = df.to_json(orient="records")
        return Response(content=json_output, media_type="application/json")
    except FileNotFoundError:
        return Response(content='{"error": "Results file not found. Run validation first."}', 
                        media_type="application/json", status_code=404)

@app.get("/api/extract-image")
def extract_image_endpoint():
    """Runs the VLM extraction on the sample image."""
    try:
        # We'll hard-code the pamphlet for this demo
        vlm_data = extract_data_from_image('pamplet 4.jpeg')
        return vlm_data
    except Exception as e:
        return Response(content=f'{{"error": "{str(e)}"}}', media_type="application/json", status_code=500)


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves the main HTML dashboard (index.html)."""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found!</h1>"