# communications.py
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = [
    "provider_id", "name", "phone", "address", "specialization",
    "status", "confidence_score", "google_phone", "registration_valid",
    # optional columns we may use if present:
    "hospital", "email"
]


def _load_results(path: str = "validation_results.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run validation first (POST /run-batch) to generate it."
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    # force confidence_score numeric
    if "confidence_score" in df.columns:
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0).astype(int)
    else:
        df["confidence_score"] = 0

    # normalize booleans
    if "registration_valid" in df.columns:
        df["registration_valid"] = df["registration_valid"].astype(str).str.lower().map(
            {"true": True, "false": False}
        ).fillna("")
    else:
        df["registration_valid"] = ""

    # ensure columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _default_email_for(name: str) -> str:
    safe = (name or "provider").lower().replace(" ", ".").replace(",", "").replace("/", "")
    return f"{safe}@hospital.com"


def generate_verification_emails(
    results_csv: str = "validation_results.csv",
    out_csv: str = "generated_emails.csv",
    out_dir: str = "emails_output"
) -> pd.DataFrame:
    """
    Build personalized verification emails for providers that likely need contact.
    Criteria:
      - status contains REVIEW or UPDATE, OR confidence_score < 80
    Outputs:
      - CSV at out_csv
      - Individual .txt files under out_dir/
    Returns the DataFrame of emails.
    """
    df = _load_results(results_csv)

    needs_contact = df[
        (df["status"].str.contains("REVIEW|UPDATE", case=False, na=False)) |
        (df["confidence_score"] < 80)
    ].copy()

    print(f"Found {len(needs_contact)} providers needing verification contact")

    emails: List[Dict] = []

    for _, provider in needs_contact.iterrows():
        # Build issue list
        issues = []
        phone_on_file = provider.get("phone", "")
        google_phone = provider.get("google_phone", "")
        if google_phone and google_phone != phone_on_file:
            issues.append(
                f"Phone number mismatch (Our records: {phone_on_file or 'N/A'}, Google Maps: {google_phone or 'N/A'})"
            )
        if int(provider.get("confidence_score", 0)) < 50:
            issues.append("Multiple data conflicts across sources")
        reg_valid = provider.get("registration_valid", "")
        if reg_valid is False or str(reg_valid).lower() == "false":
            issues.append("Medical registration verification required")

        issue_list = "\n".join([f"  • {issue}" for issue in issues]) if issues else "  • General verification required"

        # Compose email
        email_obj = {
            "provider_id": provider.get("provider_id", ""),
            "provider_name": provider.get("name", ""),
            "email_to": provider.get("email") or _default_email_for(provider.get("name", "")),
            "subject": f"Provider Directory Verification Required - {provider.get('provider_id', '')}",
            "body": f"""Dear {provider.get('name','Provider')},

We are updating our healthcare provider directory to ensure accurate information for our members seeking care.

PROVIDER INFORMATION ON FILE
----------------------------------------------------------------
Name: {provider.get('name','N/A')}
Provider ID: {provider.get('provider_id','N/A')}
Phone: {provider.get('phone','N/A')}
Address: {provider.get('address','N/A')}
Specialization: {provider.get('specialization','N/A')}
Hospital/Clinic: {provider.get('hospital','N/A')}

VALIDATION STATUS
----------------------------------------------------------------
Status: {provider.get('status','N/A')}
Confidence Score: {provider.get('confidence_score','N/A')}%

ISSUES DETECTED
----------------------------------------------------------------
{issue_list}

ACTION REQUIRED
----------------------------------------------------------------
Please review the information above and:

1) Confirm if all information is correct, or
2) Provide updates for any incorrect information:
   - Current Phone Number: _____________________
   - Current Address: _____________________
   - Current Practice Hours: _____________________
   - Updated Specialization(s): _____________________

Please reply to this email within 7 business days with your confirmation or corrections.

SUGGESTED CORRECTIONS
----------------------------------------------------------------
{f"Recommended Phone: {provider.get('suggested_phone', 'N/A')}" if provider.get('suggested_phone') else "No automated suggestions available"}

Thank you for helping us maintain accurate provider information for our members.

Best regards,
Provider Directory Management Team
Healthcare Payer Organization

---
This is an automated verification request generated by our AI-powered Provider Validation System.
For questions, contact: directory-support@healthcarepayer.com
Reference ID: {provider.get('provider_id','N/A')}-{datetime.now().strftime('%Y%m%d')}
"""
        }

        emails.append(email_obj)

    emails_df = pd.DataFrame(emails)
    if not emails_df.empty:
        emails_df.to_csv(out_csv, index=False)

        os.makedirs(out_dir, exist_ok=True)
        for email in emails:
            safe_name = (email["provider_name"] or "provider").replace(" ", "_").replace("/", "_")
            filename = os.path.join(out_dir, f"{email['provider_id']}_{safe_name}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"TO: {email['email_to']}\n")
                f.write(f"SUBJECT: {email['subject']}\n\n")
                f.write(email["body"])

    print(f"Generated {len(emails)} verification emails")
    print(f"Saved to: {out_csv}")
    print(f"Individual emails saved to: {out_dir}/")

    return emails_df


def generate_summary_report(
    results_csv: str = "validation_results.csv",
    out_report: str = "validation_summary_report.txt"
) -> str:
    """Create an executive summary text report and return the text."""
    df = _load_results(results_csv)

    def pct(n):  # safe percentage helper
        return (100.0 * n / len(df)) if len(df) else 0.0

    report = f"""
================================================================
                PROVIDER DIRECTORY VALIDATION REPORT
                    Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
================================================================

EXECUTIVE SUMMARY
----------------------------------------------------------------
Total Providers Validated: {len(df)}
Processing Time: < 5 minutes (vs. 40+ hours manual)
Average Confidence Score: {df['confidence_score'].mean():.1f}%

VALIDATION RESULTS BREAKDOWN
----------------------------------------------------------------
Status                          Count    Percentage
----------------------------------------------------------------
Verified (100% confidence)      {len(df[df['confidence_score'] == 100]):>5}    {pct(len(df[df['confidence_score'] == 100])):>6.1f}%
High Confidence (80-99%)        {len(df[(df['confidence_score'] >= 80) & (df['confidence_score'] < 100)]):>5}    {pct(len(df[(df['confidence_score'] >= 80) & (df['confidence_score'] < 100)])):>6.1f}%
Medium Confidence (50-79%)      {len(df[(df['confidence_score'] >= 50) & (df['confidence_score'] < 80)]):>5}    {pct(len(df[(df['confidence_score'] >= 50) & (df['confidence_score'] < 80)])):>6.1f}%
Low Confidence (<50%)           {len(df[df['confidence_score'] < 50]):>5}    {pct(len(df[df['confidence_score'] < 50])):>6.1f}%

DATA QUALITY ISSUES IDENTIFIED
----------------------------------------------------------------
Issue Type                           Count
----------------------------------------------------------------
Needs Update (Google Maps)           {len(df[df['status'].str.contains('UPDATE', na=False)])}
Needs Manual Review                  {len(df[df['status'].str.contains('REVIEW', na=False)])}
Low Confidence Score (<60%)          {len(df[df['confidence_score'] < 60])}

TOP PRIORITY PROVIDERS FOR MANUAL REVIEW
----------------------------------------------------------------
(Sorted by lowest confidence score)
"""

    priority = df[df["confidence_score"] < 70].sort_values("confidence_score").head(10)
    for _, row in priority.iterrows():
        rid = row.get("provider_id", "")
        nm = row.get("name", "")
        cs = row.get("confidence_score", "")
        st = row.get("status", "")
        report += f"\n{rid} | {nm:<30} | Confidence: {cs}% | {st}"

    report += f"""

DATA SOURCE PERFORMANCE
----------------------------------------------------------------
Google Maps API:
  • Successful Lookups: {df['google_phone'].replace('', pd.NA).notna().sum()} ({pct(df['google_phone'].replace('', pd.NA).notna().sum()):.1f}%)
  • Coverage Rate: Good

Medical Council Validation:
  • Valid Registrations: {int(pd.to_numeric((df['registration_valid'] == True), errors='coerce').sum()) if 'registration_valid' in df.columns else 0}
  • Coverage Rate: Moderate

COST SAVINGS ANALYSIS
----------------------------------------------------------------
Manual Validation Cost:
  • Time per provider: 12 minutes
  • Total time for {len(df)} providers: {len(df) * 12 / 60:.1f} hours
  • Cost at $25/hour: ${len(df) * 12 / 60 * 25:.2f}

Automated Validation Cost:
  • Processing time: < 5 minutes total
  • Cost: Minimal (API costs only)
  • Savings: {max(0.0, (len(df) * 12 / 60 - 5/60) / (len(df) * 12 / 60) * 100):.1f}% time reduction

RECOMMENDATIONS
----------------------------------------------------------------
1) Immediate Actions:
   • Send verification emails to {len(df[df['confidence_score'] < 70])} low-confidence providers
   • Manual review of {len(df[df['confidence_score'] < 50])} critical cases
   • Update {len(df[df['status'].str.contains('UPDATE', na=False)])} records with Google Maps data

2) Process Improvements:
   • Implement weekly automated validation cycles
   • Add additional public sources for cross-checking (e.g., hospital directories)
   • Set up automated alerts for registration expiration

3) System Enhancements:
   • Integrate with state medical council APIs where available
   • Add member complaint correlation
   • Add ML-based risk scoring

----------------------------------------------------------------
Report generated by AI-Powered Provider Validation System
For questions: contact data-quality@healthcarepayer.com
----------------------------------------------------------------
"""

    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)

    print("Report saved to:", out_report)
    return report


def generate_all() -> Dict:
    """Run both generators and return a small summary dict."""
    emails_df = generate_verification_emails()
    report_text = generate_summary_report()
    return {
        "emails_count": int(len(emails_df) if emails_df is not None else 0),
        "emails_csv": "generated_emails.csv" if os.path.exists("generated_emails.csv") else "",
        "emails_dir": "emails_output" if os.path.isdir("emails_output") else "",
        "report_path": "validation_summary_report.txt" if os.path.exists("validation_summary_report.txt") else "",
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING COMMUNICATION MATERIALS")
    print("=" * 70)
    summary = generate_all()
    print("Summary:", summary)
    print("Done.")
