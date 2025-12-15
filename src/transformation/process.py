import pandas as pd
import numpy as np
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting process.py script")

start_time = time.time()

# Load raw data
data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
input_path = os.path.join(data_dir, "movies.json")
df_movie = pd.read_json(input_path)
logger.info(f"Loaded raw data from {input_path} with {len(df_movie)} rows")

# 1: the irrelevant columns to drop
cols_to_drop = ['adult', 'imdb_id', 'original_title', 'video', 'homepage']
for col in cols_to_drop:
    # Check if the column actually exists in the DataFrame to avoid errors
    if col in cols_to_drop:
        # axis=1 refers to columns. inplace=True modifies the DataFrame directly.
        df_movie.drop(col, axis=1, inplace=True)
logger.info(f"Dropped irrelevant columns: {cols_to_drop}")

# 2. Evaluate JSON-like columns
extracted_columns = ['belongs_to_collection', 'genres', 'spoken_languages',
                     'production_countries', 'production_companies', 'credits']

# 3. Extract and clean points
for column in extracted_columns:
    if column == 'belongs_to_collection':
        # Extracts 'name' from belongs_to_collection
        df_movie[column] = df_movie[column].apply(
            lambda x: x['name'] if isinstance(x, dict) else None)
    else:
        # Extracts other columns as pipe-separated strings
        df_movie[column] = df_movie[column].apply(lambda x: "|".join(
            [item['name'] for item in x]) if isinstance(x, list) else None)

logger.info(f"Extracted and cleaned JSON-like columns: {extracted_columns}")

# 4. Convert column datatypes
numeric_cols = [
    'budget',
    'id',
    'popularity',
    'revenue',
    'runtime',
    'vote_average',
    'vote_count'
]

print(" Converting Data Types...")

# 2. Convert Numeric Columns
for col in numeric_cols:
    # errors='coerce' turns invalid parsing into NaN
    df_movie[col] = pd.to_numeric(df_movie[col], errors='coerce')

#  3. Convert Release Date
df_movie['release_date'] = pd.to_datetime(
    df_movie['release_date'], errors='coerce')

#  4. Verify the Changes
print("\nNew Data Types:")
print(df_movie.dtypes)

logger.info("Converted column data types")

print("\nSample Data Check (First 5 rows):")
print(df_movie[['title', 'release_date', 'budget', 'revenue']].head())

# 1. Replace Unrealistic Values ($0 -> NaN)
target_cols = ['budget', 'revenue', 'runtime']
# Zeros in these columns imply missing data, not that the movie cost $0 or lasted 0 minutes.
df_movie[target_cols] = df_movie[target_cols].replace(0, np.nan)

# 2. Unit Conversion (Dollars -> Millions)
df_movie['budget'] = df_movie['budget'] / 1000000
# divide by 1,000,000. NaN values remain NaN automatically.
df_movie['revenue'] = df_movie['revenue'] / 1000000

# 3. Handle Zero Votes
df_movie.loc[df_movie['vote_count'] == 0, 'vote_average'] = np.nan

# 4. Clean Text Placeholders 'overview'and 'tagline'
text_placeholders = ["", "No Data", "N/A", "Unknown"]
cols_text = ['overview', 'tagline']

df_movie[cols_text] = df_movie[cols_text].replace(text_placeholders, np.nan)

print("\n Post-Cleaning Preview ")
print(df_movie[['title', 'budget', 'revenue',
      'runtime', 'vote_count', 'vote_average']].head())

print("\n Missing Value Count (After setting 0 to NaN) ")
print(df_movie[['budget', 'revenue', 'runtime']].isna().sum())

print(f"Rows before filtering: {df_movie.shape[0]}")

# Step 7: Remove Duplicates & Missing Identifiers
df_movie.dropna(subset=['id', 'title'], inplace=True)

# 2. Remove Duplicate IDs (Keep the first occurrence)
df_movie.drop_duplicates(subset=['id'], inplace=True)

print(f"Rows after removing duplicates/missing IDs: {df_movie.shape[0]}")

# Step 8: Quality Threshold (Drop "Empty" Rows)
df_movie.dropna(thresh=10, inplace=True)

print(f"Rows after threshold check (>=10 columns): {df_movie.shape[0]}")

# Step 9: Filter by Status & Clean Up
df_movie = df_movie[df_movie['status'] == 'Released'].copy()

# 2. Drop the 'status' column (Redundant now, as all are 'Released')
df_movie.drop(columns=['status'], inplace=True)

print(f"Final Row Count (Released only): {df_movie.shape[0]}")

logger.info(f"Filtered to {df_movie.shape[0]} released movies")

#  Final Dataset Verification
print("\n Final Dataset Info ")
df_movie.info()

print("\n Sample of Final Data ")
print(df_movie[['title', 'release_date', 'budget', 'revenue']].head())

# 1. Rename columns to match the target schema
rename_map = {
    'budget': 'budget_musd',
    'revenue': 'revenue_musd',
    'id': 'id'
}
df_movie.rename(columns=rename_map, inplace=True)

# 2. Define the Target Column Order
target_order = [
    'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
    'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
    'production_countries', 'vote_count', 'vote_average', 'popularity', 'runtime',
    'overview', 'spoken_languages', 'poster_path',
    # These will be created as NaN if missing
    'cast', 'cast_size', 'director', 'crew_size', 'credits'
]

print(" Reordering Columns ")

# 3. Reorder and Enforce Schema
# .reindex() is safer than df_movie[target_order] because it handles missing columns gracefully
df_movie = df_movie.reindex(columns=target_order)

# 4. Reset Index
df_movie.reset_index(drop=True, inplace=True)

#  Final Verification
print("\nFinal Dataframe Shape:", df_movie.shape)
print("\nColumn List:")
print(df_movie.columns.tolist())

print("\nData Preview:")
print(df_movie[['id', 'title', 'budget_musd', 'genres']].head())

# Export Cleaned Data to CSV
# 1. Define the filename
data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
os.makedirs(data_dir, exist_ok=True)
output_filename = os.path.join(data_dir, "movies_clean.csv")

# 2. Export to CSV
print(f" Saving data to {output_filename} ")
df_movie.to_csv(output_filename, index=False)

# 3. Check if file exists and print its size to confirm success
if os.path.exists(output_filename):
    file_size = os.path.getsize(output_filename) / 1024
    print(f"Success: File saved. Size: {file_size:.2f} KB")

    # read back the first line to prove it works
    test_load = pd.read_csv(output_filename, nrows=1)
    print("\nFile Header Preview:")
    print(test_load.columns.tolist())
else:
    print("Error: File was not created.")
    logger.error("Failed to create output file")

end_time = time.time()
logger.info(
    f"process.py completed successfully in {end_time - start_time:.2f} seconds")
