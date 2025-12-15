import os
import requests
from dotenv import load_dotenv
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting fetch_data.py script")

start_time = time.time()

# Load API key from .env file
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)
api_key = os.getenv("API_KEY")
if not api_key:
    logger.error("API_KEY not found in .env file")
    raise ValueError("API_KEY not found.")
logger.info("API key loaded successfully")

# Movie IDs to fetch
movie_ids = [
    0, 299534, 19995, 140607, 299536, 597, 135397, 420818,
    24428, 168259, 99861, 284054, 12445, 181808, 330457,
    351286, 109445, 321612, 260513
]
selected_movie_ids = []
logger.info(
    f"Starting to fetch data for {len([id for id in movie_ids if id != 0])} movies")
for movie_id in movie_ids:
    if movie_id == 0:
        continue
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?append_to_response=credits&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        selected_movie_ids.append(response.json())
        logger.info(f"Successfully fetched movie ID {movie_id}")
    else:
        logger.warning(
            f"Failed to fetch movie {movie_id}: {response.status_code}")
logger.info(f"Fetched data for {len(selected_movie_ids)} movies")

# Store data as a Pandas DataFrame
df_movie = pd.DataFrame(selected_movie_ids)
logger.info(
    f"Created DataFrame with {len(df_movie)} rows and {len(df_movie.columns)} columns")

# Save raw data
data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
os.makedirs(data_dir, exist_ok=True)
output_path = os.path.join(data_dir, "movies.json")
df_movie.to_json(output_path, orient='records')
logger.info(f"Saved raw data to {output_path}")

end_time = time.time()
logger.info(
    f"fetch_data.py completed successfully in {end_time - start_time:.2f} seconds")
