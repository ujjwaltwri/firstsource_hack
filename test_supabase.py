import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()  # Loads .env from current directory

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

print(f"SUPABASE_URL: {url}")
if not url or not key:
    print("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
else:
    client = create_client(url, key)
    print("Connected to Supabase successfully!")
