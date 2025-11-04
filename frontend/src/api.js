const API_BASE = "http://127.0.0.1:8000";

export async function uploadAndValidateCsv(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/validate-csv`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`CSV validate failed: ${res.status}`);
  return res.json(); // { rows: [...] }
}

export async function processOcr(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/ocr`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`OCR failed: ${res.status}`);
  return res.json(); // OCR json
}

export async function health() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
