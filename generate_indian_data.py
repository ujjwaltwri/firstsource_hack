import pandas as pd
import random
from datetime import datetime, timedelta

print("🇮🇳 Generating Smart Synthetic Indian Healthcare Provider Dataset...")
print("This data is designed to demonstrate your validation system effectively!\n")

# Indian names database
first_names_male = ["Rajesh", "Amit", "Suresh", "Vikram", "Arun", "Karthik", "Ramesh", "Sanjay", 
                    "Praveen", "Ajay", "Manoj", "Ravi", "Deepak", "Anand", "Ashok", "Rahul", "Nitin"]
first_names_female = ["Priya", "Anjali", "Neha", "Kavita", "Sunita", "Meera", "Lakshmi", 
                      "Divya", "Pooja", "Sneha", "Ritu", "Swati", "Nisha", "Preeti", "Asha"]
last_names = ["Kumar", "Sharma", "Patel", "Singh", "Reddy", "Iyer", "Gupta", "Rao", 
              "Mehta", "Desai", "Verma", "Agarwal", "Joshi", "Nair", "Pillai", "Kulkarni", "Menon"]

# Cities and their details
cities_data = {
    "Bangalore": {"state": "Karnataka", "std_code": "080", "pincode_prefix": "560"},
    "Delhi": {"state": "Delhi", "std_code": "011", "pincode_prefix": "110"},
    "Mumbai": {"state": "Maharashtra", "std_code": "022", "pincode_prefix": "400"},
    "Chennai": {"state": "Tamil Nadu", "std_code": "044", "pincode_prefix": "600"},
    "Hyderabad": {"state": "Telangana", "std_code": "040", "pincode_prefix": "500"},
    "Pune": {"state": "Maharashtra", "std_code": "020", "pincode_prefix": "411"},
    "Kolkata": {"state": "West Bengal", "std_code": "033", "pincode_prefix": "700"},
    "Jaipur": {"state": "Rajasthan", "std_code": "0141", "pincode_prefix": "302"},
    "Ahmedabad": {"state": "Gujarat", "std_code": "079", "pincode_prefix": "380"},
    "Lucknow": {"state": "Uttar Pradesh", "std_code": "0522", "pincode_prefix": "226"},
    "Bikaner": {"state": "Rajasthan", "std_code": "0151", "pincode_prefix": "334"},
}

# Hospitals by city (mix of real chains and generic names)
hospitals = {
    "Bangalore": ["Apollo Hospital Bangalore", "Manipal Hospital Whitefield", "Fortis Hospital Bannerghatta", 
                  "Columbia Asia Hebbal", "Narayana Health City", "City Care Clinic", "Wellness Hospital"],
    "Delhi": ["AIIMS Delhi", "Max Super Speciality Saket", "Fortis Hospital Vasant Kunj", 
              "Apollo Hospital Sarita Vihar", "BLK Hospital", "Metro Hospital", "Capital Clinic"],
    "Mumbai": ["Lilavati Hospital", "Breach Candy Hospital", "Hinduja Hospital Khar", 
               "Kokilaben Dhirubhai Ambani", "Fortis Hospital Mulund", "Suburban Clinic", "Marine Drive Hospital"],
    "Chennai": ["Apollo Hospital Greams Road", "MIOT International", "Fortis Malar Hospital", 
                "Kauvery Hospital Alwarpet", "SIMS Hospital Vadapalani", "City Medical Center", "Beach Care Hospital"],
    "Hyderabad": ["Apollo Hospital Jubilee Hills", "Yashoda Hospital Secunderabad", "KIMS Hospital Begumpet", 
                  "Care Hospital Banjara Hills", "Continental Hospital Gachibowli", "Sunrise Clinic", "Metro Care"],
    "Pune": ["Ruby Hall Clinic", "Sahyadri Hospital Deccan", "Aditya Birla Hospital", 
             "Deenanath Mangeshkar Hospital", "Columbia Asia Kharadi", "Koregaon Park Clinic", "Model Hospital"],
    "Kolkata": ["Apollo Gleneagles", "AMRI Hospital Salt Lake", "Fortis Hospital Anandapur", 
                "Peerless Hospital", "Medica Superspecialty", "Park Street Clinic", "Alipore Medical Center"],
    "Jaipur": ["SMS Hospital", "Fortis Escorts Hospital", "Narayana Multispeciality Hospital", 
               "Eternal Heart Care", "CK Birla Hospital RBH", "Malviya Nagar Clinic", "Pink City Hospital"],
    "Ahmedabad": ["Sterling Hospital Memnagar", "Apollo Hospital Bhat", "Shalby Hospital SG Highway", 
                  "SAL Hospital Drive In", "HCG Cancer Centre", "Satellite Clinic", "West Zone Hospital"],
    "Lucknow": ["SGPGIMS", "King George Medical University", "Apollo Medics Hospital", 
                "Sahara Hospital", "Medanta Hospital Lucknow", "Gomti Nagar Clinic", "Hazratganj Medical"],
    "Bikaner": {"state": "Rajasthan", "std_code": "0151", "pincode_prefix": "334"},
}

# Add Bikaner hospitals
hospitals["Bikaner"] = ["Royal Hospital", "PBM Hospital", "Bikaner Medical Center", 
                        "Gandhi Nagar Clinic", "Sadul Hospital", "City Care Bikaner"]

# Specializations with realistic distribution
specializations_weighted = [
    ("General Physician", "MBBS, MD (Medicine)", 35),
    ("Gynecologist", "MBBS, MS (Obstetrics & Gynaecology)", 12),
    ("Pediatrician", "MBBS, MD (Pediatrics)", 12),
    ("Cardiologist", "MBBS, MD, DM (Cardiology)", 8),
    ("Orthopedic Surgeon", "MBBS, MS (Orthopedics)", 8),
    ("ENT Specialist", "MBBS, DLO", 8),
    ("Dermatologist", "MBBS, MD (Dermatology)", 6),
    ("Neurologist", "MBBS, MD, DM (Neurology)", 4),
    ("Ophthalmologist", "MBBS, MS (Ophthalmology)", 4),
    ("Dentist", "BDS, MDS", 3),
]

# Areas/localities by city
localities = {
    "Bangalore": ["MG Road", "Indiranagar", "Koramangala", "Jayanagar", "Whitefield", "HSR Layout", "BTM Layout"],
    "Delhi": ["Saket", "Dwarka", "Rohini", "Lajpat Nagar", "Vasant Kunj", "Connaught Place", "Greater Kailash"],
    "Mumbai": ["Andheri West", "Bandra", "Powai", "Thane", "Borivali", "Churchgate", "Worli"],
    "Chennai": ["T Nagar", "Anna Nagar", "Velachery", "Adyar", "Porur", "OMR", "Nungambakkam"],
    "Hyderabad": ["Banjara Hills", "Jubilee Hills", "KPHB", "Gachibowli", "Secunderabad", "Kukatpally", "Madhapur"],
    "Pune": ["Koregaon Park", "Kothrud", "Aundh", "Viman Nagar", "Hinjewadi", "Pimpri", "Deccan"],
    "Kolkata": ["Park Street", "Salt Lake", "Alipore", "Ballygunge", "Howrah", "New Town", "Behala"],
    "Jaipur": ["Malviya Nagar", "Vaishali Nagar", "C Scheme", "Mansarovar", "Jagatpura", "Raja Park", "Tonk Road"],
    "Ahmedabad": ["Satellite", "Navrangpura", "Vastrapur", "Prahlad Nagar", "SG Highway", "Bodakdev", "Maninagar"],
    "Lucknow": ["Gomti Nagar", "Hazratganj", "Aliganj", "Indira Nagar", "Alambagh", "Mahanagar", "Ashiyana"],
    "Bikaner": ["Gandhi Nagar", "Karni Nagar", "Sadul Ganj", "Jail Road", "Station Road", "Ganga Shahar"],
}

# State medical council codes
smc_codes = {
    "Karnataka": "KMC",
    "Delhi": "DMC",
    "Maharashtra": "MMC",
    "Tamil Nadu": "TNMC",
    "Telangana": "TSMC",
    "West Bengal": "WBMC",
    "Rajasthan": "RMC",
    "Gujarat": "GMC",
    "Uttar Pradesh": "UPMC",
}

# Error types for demonstration
ERROR_TYPES = [
    "CORRECT_DATA",           # 60% - Everything matches
    "OUTDATED_PHONE",         # 15% - Phone changed
    "MOVED_ADDRESS",          # 10% - Clinic relocated
    "MULTIPLE_CONFLICTS",     # 8%  - Phone AND address wrong
    "HOSPITAL_CHANGED",       # 5%  - Switched hospitals
    "REGISTRATION_ISSUE",     # 2%  - License/registration problem
]

def get_error_type():
    """Randomly assign error type with realistic distribution"""
    rand = random.random()
    if rand < 0.60:
        return "CORRECT_DATA"
    elif rand < 0.75:
        return "OUTDATED_PHONE"
    elif rand < 0.85:
        return "MOVED_ADDRESS"
    elif rand < 0.93:
        return "MULTIPLE_CONFLICTS"
    elif rand < 0.98:
        return "HOSPITAL_CHANGED"
    else:
        return "REGISTRATION_ISSUE"

def generate_indian_phone(std_code, is_correct=True):
    """Generate Indian phone number"""
    if is_correct:
        # Current active numbers
        return f"{std_code}-{random.randint(40000000, 99999999)}"
    else:
        # Old/outdated numbers
        return f"{std_code}-{random.randint(20000000, 39999999)}"

def generate_mobile(is_correct=True):
    """Generate Indian mobile number"""
    if is_correct:
        # Active series: 7, 8, 9
        prefix = random.choice([7, 8, 9])
        return f"+91-{prefix}{random.randint(000000000, 999999999)}"
    else:
        # Older disconnected numbers
        return f"+91-6{random.randint(000000000, 999999999)}"

def generate_registration_number(state, year, is_valid=True):
    """Generate state medical council registration"""
    code = smc_codes[state]
    if is_valid:
        reg_num = random.randint(10000, 99999)
        return f"{code}/{reg_num}/{year}"
    else:
        # Invalid format or expired
        return f"{code}/EXPIRED/{year-20}"

def get_specialization():
    """Get specialization based on realistic distribution"""
    total_weight = sum(w for _, _, w in specializations_weighted)
    rand_weight = random.randint(1, total_weight)
    
    cumulative = 0
    for spec, qual, weight in specializations_weighted:
        cumulative += weight
        if rand_weight <= cumulative:
            return spec, qual
    
    return specializations_weighted[0][:2]

def generate_provider(provider_id):
    """Generate a single provider record with smart error injection"""
    
    # Random gender
    is_male = random.choice([True, False])
    first_name = random.choice(first_names_male if is_male else first_names_female)
    last_name = random.choice(last_names)
    name = f"Dr. {first_name} {last_name}"
    
    # Random city
    city = random.choice(list(cities_data.keys()))
    city_info = cities_data[city]
    state = city_info["state"]
    
    # Random hospital
    hospital = random.choice(hospitals[city])
    
    # Random locality
    locality = random.choice(localities[city])
    
    # Random specialization (weighted)
    specialization, qualification = get_specialization()
    
    # Registration details
    reg_year = random.randint(2008, 2023)
    
    # Determine error type for this provider
    error_type = get_error_type()
    
    # Generate data based on error type
    is_phone_correct = error_type not in ["OUTDATED_PHONE", "MULTIPLE_CONFLICTS"]
    is_address_correct = error_type not in ["MOVED_ADDRESS", "MULTIPLE_CONFLICTS"]
    is_hospital_correct = error_type != "HOSPITAL_CHANGED"
    is_registration_valid = error_type != "REGISTRATION_ISSUE"
    
    # Phone numbers
    phone = generate_indian_phone(city_info["std_code"], is_phone_correct)
    mobile = generate_mobile(is_phone_correct)
    
    # Address
    street_num = random.randint(1, 999)
    pincode = f"{city_info['pincode_prefix']}{random.randint(100, 999):03d}"
    
    if is_address_correct:
        address = f"{street_num}, {locality}, {city}, {state} - {pincode}"
    else:
        # Old address (moved)
        old_locality = random.choice([l for l in localities[city] if l != locality])
        address = f"{street_num}, {old_locality}, {city}, {state} - {pincode}"
    
    # Hospital
    if not is_hospital_correct:
        # Switched to different hospital
        old_hospital = random.choice([h for h in hospitals[city] if h != hospital])
        hospital = old_hospital
    
    # Registration
    registration_number = generate_registration_number(state, reg_year, is_registration_valid)
    
    # Email
    email_domain = hospital.split()[0].lower().replace("'", "")
    email = f"dr.{first_name.lower()}.{last_name.lower()}@{email_domain}.com"
    
    # Consultation fee (varies by specialization)
    fee_ranges = {
        "General Physician": ["₹300-500", "₹400-600"],
        "Gynecologist": ["₹500-800", "₹600-1000"],
        "Cardiologist": ["₹1000-1500", "₹1200-2000"],
        "Orthopedic Surgeon": ["₹800-1200", "₹1000-1500"],
    }
    fee_range = random.choice(fee_ranges.get(specialization, ["₹500-800", "₹600-1000"]))
    
    # Timings
    timings_options = [
        "Mon-Sat: 9 AM - 5 PM",
        "Mon-Fri: 10 AM - 6 PM",
        "Tue-Sun: 11 AM - 7 PM",
        "Mon-Sat: 8 AM - 2 PM, 5 PM - 8 PM",
        "Mon-Fri: 9 AM - 1 PM, 4 PM - 8 PM"
    ]
    timings = random.choice(timings_options)
    
    # Languages
    languages = ["English", "Hindi"]
    state_languages = {
        "Karnataka": "Kannada",
        "Tamil Nadu": "Tamil",
        "Maharashtra": "Marathi",
        "West Bengal": "Bengali",
        "Gujarat": "Gujarati",
        "Telangana": "Telugu",
        "Rajasthan": "Hindi",
        "Uttar Pradesh": "Hindi",
    }
    if state in state_languages:
        languages.append(state_languages[state])
    
    return {
        "provider_id": provider_id,
        "name": name,
        "registration_number": registration_number,
        "qualification": qualification,
        "specialization": specialization,
        "hospital": hospital,
        "address": address,
        "phone": phone,
        "mobile": mobile,
        "email": email,
        "city": city,
        "state": state,
        "consultation_fee": fee_range,
        "timings": timings,
        "languages": ", ".join(languages),
        "expected_error_type": error_type,  # For testing - shows what error was injected
    }

# Add Dr. Kalaiarasan as the first real entry (the one you have)
real_doctor = {
    "provider_id": "P001",
    "name": "Dr. A. Kalaiarasan",
    "registration_number": "RMC/11223/2012",
    "qualification": "MBBS, DLO",
    "specialization": "ENT Specialist",
    "hospital": "Royal Hospital",
    "address": "Gandhi Nagar, Bikaner, Rajasthan - 334001",
    "phone": "0151-2304455",  # Slightly different from pamphlet
    "mobile": "+91-9876543210",
    "email": "info@royalhospital.net.in",
    "city": "Bikaner",
    "state": "Rajasthan",
    "consultation_fee": "₹500-800",
    "timings": "Mon-Sat: 10 AM - 6 PM",
    "languages": "English, Hindi, Rajasthani",
    "expected_error_type": "OUTDATED_PHONE",  # Phone in CSV vs pamphlet
}

# Generate 199 more synthetic providers
providers = [real_doctor]
for i in range(2, 201):
    provider = generate_provider(f"P{i:03d}")
    providers.append(provider)

# Create DataFrame
df = pd.DataFrame(providers)

# Save to CSV
df.to_csv('input_providers_india.csv', index=False)

# Generate summary
print(f"✓ Generated {len(df)} Indian providers\n")
print(f"📊 Dataset Composition:")
print(f"  • Cities covered: {df['city'].nunique()}")
print(f"  • States: {df['state'].nunique()}")
print(f"  • Specializations: {df['specialization'].nunique()}")
print(f"  • Hospitals: {df['hospital'].nunique()}\n")

print(f"🎯 Error Distribution (for validation testing):")
error_counts = df['expected_error_type'].value_counts()
for error_type, count in error_counts.items():
    percentage = (count/len(df))*100
    print(f"  • {error_type}: {count} ({percentage:.1f}%)")

print(f"\n📁 Saved to: input_providers_india.csv")
print(f"\n🔍 First 5 records:")
display_cols = ['provider_id', 'name', 'specialization', 'city', 'phone', 'expected_error_type']
print(df[display_cols].head().to_string(index=False))

print(f"\n💡 Note: 'expected_error_type' column shows what validation errors to expect")
print(f"   Remove this column before final submission - it's just for testing!")