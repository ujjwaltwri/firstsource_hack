# supabase_client.py
import os
from supabase import create_client, Client

_supabase: Client | None = None

def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")
        _supabase = create_client(url, key)
    return _supabase
