import os
import time
import logging
import requests
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TMDBDataFetcher:
    """A production-grade fetcher for TMDB movie data."""
    
    def __init__(self, env_path: str = "../../.env"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_config(env_path)
        self.session = self._get_session()
        self.base_url = "https://api.themoviedb.org/3/movie"

    def _load_config(self, env_path: str):
        """Loads environment variables and sets up paths."""
        load_dotenv(os.path.join(os.path.dirname(__file__), env_path))
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            self.logger.error("API_KEY missing from environment.")
            raise ValueError("API_KEY not found.")
            
        # Define paths relative to this class
        self.raw_data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def _get_session(self) -> requests.Session:
        """Creates a requests session with built-in retry logic."""
        session = requests.Session()
        # Retry strategy: 3 retries, backoff factor handles 429 and 5xx errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # Wait 1s, 2s, 4s...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_single_movie(self, movie_id: int) -> Optional[dict]:
        """Fetches a single movie with error handling and timeouts."""
        if movie_id == 0:
            return None
            
        params = {
            "api_key": self.api_key,
            "append_to_response": "credits"
        }
        
        try:
            # Added timeout (10s) to prevent hanging
            response = self.session.get(
                f"{self.base_url}/{movie_id}", 
                params=params, 
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"ID {movie_id} failed: {e.response.status_code}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching ID {movie_id}: {str(e)}")
        return None

    def run_pipeline(self, movie_ids: List[int]) -> pd.DataFrame:
        """Executes the full extraction and saving process."""
        start_time = time.time()
        results = []
        
        self.logger.info(f"Starting extraction for {len(movie_ids)} IDs...")
        
        for mid in movie_ids:
            data = self.fetch_single_movie(mid)
            if data:
                results.append(data)
                self.logger.info(f"Fetched: {data.get('title', mid)}")
            
            # Compliance with TMDB rate limits (approx 4 req/sec)
            time.sleep(0.25)

        df = pd.DataFrame(results)
        self._save_data(df)
        
        duration = time.time() - start_time
        self.logger.info(f"Pipeline complete. {len(df)} movies saved in {duration:.2f}s")
        return df

    def _save_data(self, df: pd.DataFrame):
        """Saves the DataFrame to the raw data directory."""
        output_path = os.path.join(self.raw_data_dir, "movies.json")
        df.to_json(output_path, orient='records', indent=4)
        self.logger.info(f"Data persisted to {output_path}")

# --- Execution Block ---
if __name__ == "__main__":
    # Setup logging configuration once at the entry point
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ids = [0, 299534, 19995, 140607, 299536, 597, 135397, 420818, 24428, 168259, 99861, 284054, 12445, 181808, 330457, 351286, 109445, 321612, 260513]
    
    fetcher = TMDBDataFetcher()
    movie_df = fetcher.run_pipeline(ids)