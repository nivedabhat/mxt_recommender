from pyairtable import Table
import pandas as pd
import os

# ----------------------------
# Config
# ----------------------------
PAT = os.environ.get("AIRTABLE_PAT")  # <- read from environment variable
BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
TABLE_NAME = "Quiz and Product Table - Sandbox"
OUTPUT_PATH = "data/raw/quiz_product_table.csv"

# ----------------------------
# Ensure output folder exists
# ----------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ----------------------------
# Connect to Airtable
# ----------------------------
table = Table(PAT, BASE_ID, TABLE_NAME)

# ----------------------------
# Pull all records with pagination
# ----------------------------
all_records = []
offset = None

while True:
    response = table.all(page_size=100, offset=offset)
    if not response:
        break
    all_records.extend(response)
    offset = response[-1]['id'] if len(response) == 100 else None
    if not offset:
        break

# Convert to DataFrame
df = pd.DataFrame([r["fields"] for r in all_records])

# Save CSV
df.to_csv(OUTPUT_PATH, index=False)
print(f"Pulled {len(df)} rows successfully! Saved to {OUTPUT_PATH}")
print("Preview of first 5 rows:")
print(df.head())


