import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import json
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

logger.info("Starting analysis.py script")

start_time = time.time()

# Load cleaned data
data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
input_path = os.path.join(data_dir, "movies_clean.csv")
df_movie = pd.read_csv(input_path)
logger.info(f"Loaded cleaned data from {input_path} with {len(df_movie)} rows")

# Profit = Revenue - Budget
df_movie['profit_musd'] = df_movie['revenue_musd'] - df_movie['budget_musd']

# ROI = Revenue / Budget
df_movie['roi'] = df_movie['revenue_musd'] / df_movie['budget_musd']

print("KPIs Calculated: ")

logger.info("Calculated KPIs: profit_musd and roi")

# RANKING LOGIC


def print_ranking(title, df, sort_col, ascending=False, top_n=1, cols_to_show=None):
    if cols_to_show is None:
        cols_to_show = ['title', 'budget_musd', 'revenue_musd',
                        'profit_musd', 'roi', 'vote_average']
    if sort_col not in df.columns:
        print(f"Skipping {title}: Column {sort_col} not found.")
        return
    # Sort
    ranked = df.sort_values(by=sort_col, ascending=ascending).head(top_n)
    print(f"\n {title} ")
    print(ranked[cols_to_show].to_string(index=False))

    # Tabulted Output
    print(f"\n{title}")
    print(tabulate(
        ranked[cols_to_show],
        headers=cols_to_show,
        tablefmt="github",
        showindex=False
    ))


# 1. Highest Revenue
print_ranking("Highest Revenue", df_movie, 'revenue_musd')

# 2. Highest Budget
print_ranking("Highest Budget", df_movie, 'budget_musd')

# 3. Highest Profit
print_ranking("Highest Profit", df_movie, 'profit_musd')

# 4. Lowest Profit
print_ranking("Lowest Profit", df_movie, 'profit_musd', ascending=True)

# 5. Highest ROI (Budget >= 10M)
high_budget_mask = df_movie['budget_musd'] >= 10
print_ranking("Highest ROI (Budget >= 10M)", df_movie[high_budget_mask], 'roi')

# 6. Lowest ROI (Budget >= 10M)
print_ranking("Lowest ROI (Budget >= 10M)",
              df_movie[high_budget_mask], 'roi', ascending=True)

# 7. Most Voted Movies
print_ranking("Most Voted Movies", df_movie, 'vote_count',
              cols_to_show=['title', 'vote_count', 'vote_average'])

# RATING RANKINGS

# 8. Highest Rated (Votes >= 10)
valid_votes_mask = df_movie['vote_count'] >= 10
print_ranking("Highest Rated (Votes >= 10)", df_movie[valid_votes_mask], 'vote_average', cols_to_show=[
              'title', 'vote_average', 'vote_count'])

# 9. Lowest Rated (Votes >= 10)
print_ranking("Lowest Rated (Votes >= 10)", df_movie[valid_votes_mask], 'vote_average', ascending=True, cols_to_show=[
              'title', 'vote_average', 'vote_count'])

# 10. Most Popular
print_ranking("Most Popular Movies", df_movie, 'popularity',
              cols_to_show=['title', 'popularity', 'genres'])

logger.info("Completed movie rankings")

# extract cast and director


def get_cast(credits_data):
    # Extracts top 5 actors from the credits dictionary.
    if isinstance(credits_data, dict) and 'cast' in credits_data:
        actors = [x['name'] for x in credits_data['cast'][:5]]
        return "|".join(actors)
    return np.nan


def get_director(credits_data):
    # Extracts the Director's name from the credits dictionary.
    if isinstance(credits_data, dict) and 'crew' in credits_data:
        directors = [x['name']
                     for x in credits_data['crew'] if x.get('job') == 'Director']
        return "|".join(directors)
    return np.nan


print(" Populating Cast & Director ")
# We apply this to the raw 'credits' column
df_movie['cast'] = df_movie['credits'].apply(get_cast)
df_movie['director'] = df_movie['credits'].apply(get_director)

print("Sample Extraction:")
print(df_movie[['title', 'director', 'cast']].head(3))

# UDF - User-Defined Function (UDF)


def rank_movies(df, criteria_col, ascending=False, top_n=5, show_cols=None):
    if show_cols is None:
        show_cols = ['title', criteria_col, 'genres', 'director', 'cast']

    # Error handling: check if column exists
    if criteria_col not in df.columns:
        return f"Error: Column '{criteria_col}' not found."

    # Sort and slice
    ranked_df = df.sort_values(
        by=criteria_col, ascending=ascending).head(top_n)

    return ranked_df[show_cols]


# 1. LOAD CLEAN DATA
df_clean = df_movie

# 2. DROP EXISTING 'CREDITS'
if 'credits' in df_clean.columns:
    df_clean.drop(columns=['credits'], inplace=True)

# 3. PREPARE CREDITS DATA
data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
input_path = os.path.join(data_dir, "movies.json")
with open(input_path, 'r') as f:
    selected_movie_ids = json.load(f)
df_credits = pd.DataFrame(selected_movie_ids)[['id', 'credits']]

# 4. MERGE
df_movie = pd.merge(df_clean, df_credits, on='id', how='left')

# 5. DEFINE EXTRACTION FUNCTIONS


def get_cast(credits_data):
    if isinstance(credits_data, dict) and 'cast' in credits_data:
        actors = [x['name'] for x in credits_data['cast'][:5]]
        return "|".join(actors)
    return ""


def get_director(credits_data):
    if isinstance(credits_data, dict) and 'crew' in credits_data:
        directors = [x['name']
                     for x in credits_data['crew'] if x.get('job') == 'Director']
        return "|".join(directors)
    return ""


print("Repopulating Cast & Director")

# 6. APPLY EXTRACTION
df_movie['cast'] = df_movie['credits'].apply(get_cast)
df_movie['director'] = df_movie['credits'].apply(get_director)

# Verification
print(df_movie[['title', 'cast', 'director']].head())

# Search 1
print("\nSearch 1: Bruce Willis (Sci-Fi / Action) ")
# 1. Create the Boolean Mask
search_1 = (
    df_movie['genres'].str.contains("Science Fiction", na=False) &
    df_movie['genres'].str.contains("Action", na=False) &
    df_movie['cast'].str.contains("Bruce Willis", na=False)
)

# 2. Apply Filter
bruce_movies = df_movie[search_1]

# 3. Rank Results
if not bruce_movies.empty:
    print(rank_movies(bruce_movies, criteria_col='vote_average', ascending=False))
else:
    print("No movies found for this criteria.")

# Search 2
print("\nSearch 2: Uma Thurman & Quentin Tarantino ")

# 1. Create the Boolean Mask
search_2 = (
    df_movie['cast'].str.contains("Uma Thurman", na=False) &
    df_movie['director'].str.contains("Quentin Tarantino", na=False)
)

# 2. Apply Filter
qt_movies = df_movie[search_2]

# 3. Rank Results (Sorted by runtime ascending)
if not qt_movies.empty:
    print(rank_movies(qt_movies, criteria_col='runtime', ascending=True))
else:
    print("No movies found for this criteria.")

# Re-calculate KPI
df_movie['profit_musd'] = df_movie['revenue_musd'] - df_movie['budget_musd']
df_movie['roi'] = df_movie['revenue_musd'] / df_movie['budget_musd']

# 1. Create a "Franchise" Flag
# If belongs_to_collection has text, it's a Franchise. else it's Standalone.
df_movie['is_franchise'] = df_movie['belongs_to_collection'].notna()

# 2. Group and Aggregate
franchise_stats = df_movie.groupby('is_franchise').agg({
    'revenue_musd': 'mean',
    'roi': 'median',
    'budget_musd': 'mean',
    'popularity': 'mean',
    'vote_average': 'mean',
    'title': 'count'          # To see how many movies are in each category
}).rename(columns={'title': 'movie_count'})

# Rename the index for clarity
franchise_stats.index = ['Standalone', 'Franchise']

print("Category", franchise_stats.round(2))

print("\n 5. Most Successful Franchises ")

# Group by Collection Name
franchise_ranking = df_movie.groupby('belongs_to_collection').agg({
    'title': 'count',
    'budget_musd': ['sum', 'mean'],
    'revenue_musd': ['sum', 'mean'],
    'vote_average': 'mean'
})

# Flatten the MultiIndex columns (e.g., ('budget_musd', 'sum') -> 'budget_sum')
franchise_ranking.columns = ['_'.join(col).strip()
                             for col in franchise_ranking.columns.values]
franchise_ranking.rename(columns={'title_count': 'total_movies'}, inplace=True)

# Sort by Total Revenue (Descending)
top_franchises = franchise_ranking.sort_values(
    by='revenue_musd_sum', ascending=False)

print(top_franchises.head(5).round(2))

print("\n 6. Most Successful Directors ")

# Group by Director
director_ranking = df_movie[df_movie['director'] != ""].groupby('director').agg({
    'title': 'count',
    'revenue_musd': 'sum',
    'vote_average': 'mean'
})

director_ranking.rename(
    columns={'title': 'total_movies', 'revenue_musd': 'total_revenue'}, inplace=True)

# Sort by Total Revenue (Descending)
top_directors = director_ranking.sort_values(
    by='total_revenue', ascending=False)

print(top_directors.head(5).round(2))

logger.info("Completed franchise and director analysis")

end_time = time.time()
logger.info(
    f"analysis.py completed successfully in {end_time - start_time:.2f} seconds")
