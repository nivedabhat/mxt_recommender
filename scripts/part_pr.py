import pandas as pd

# Load your cleaned CSV
df = pd.read_csv("data/processed/quiz_product_table_cleaned.csv")

# Standardize column names: strip spaces and replace spaces with underscores
df.columns = df.columns.str.strip().str.replace(" ", "_")

# List of product eligibility columns
product_cols = [col for col in df.columns if "PR_-_Part" in col and "Eligible" in col]

exploded_dfs = []

for col in product_cols:
    temp_df = df.copy()
    # Convert string tuples/lists into actual lists
    temp_df[col] = temp_df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("(") else x)
    # Explode so each product gets its own row
    temp_df = temp_df.explode(col)
    # Rename product column to a common name
    temp_df = temp_df.rename(columns={col: "Product_ID"})
    # Track which part the product came from
    temp_df["Part_Origin"] = col
    # Ensure required columns exist for groupby
    required_cols = ["Product_ID", "Question_Category", "Question_Category_smoothed_prob", "Part_Origin"] + \
                    [c for c in temp_df.columns if c not in ["Product_ID", "Question_Category", "Question_Category_smoothed_prob", "Part_Origin"]]
    # Filter columns that actually exist to avoid KeyError
    required_cols = [c for c in required_cols if c in temp_df.columns]
    temp_df = temp_df[required_cols]
    exploded_dfs.append(temp_df)

# Concatenate all exploded parts
df_exploded = pd.concat(exploded_dfs, ignore_index=True)

# Deduplicate products across all parts per Question_Category
# Average the probability if duplicates exist
df_cleaned = df_exploded.groupby(["Product_ID", "Question_Category"], as_index=False).agg({
    "Question_Category_smoothed_prob": "mean",
    "Part_Origin": lambda x: ','.join(sorted(x.unique())),
    # Keep first occurrence of other columns
    **{c: 'first' for c in df_exploded.columns if c not in ["Product_ID", "Question_Category", "Question_Category_smoothed_prob", "Part_Origin"]}
})

# Reset index
df_cleaned.reset_index(drop=True, inplace=True)

# Save the final CSV
df_cleaned.to_csv("data/processed/quiz_product_table_final_with_parts.csv", index=False)
print("✅ Final cleaned CSV saved with part origins tracked!")

