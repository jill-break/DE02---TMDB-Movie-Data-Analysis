import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

class MovieTransformer:
    """A production-grade transformer for TMDB raw movie data."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processed_data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def flatten_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes nested JSON structures into flat strings or specific keys."""
        df = df.copy()
        
        # 1. Handle Collection (Dictionary)
        if 'belongs_to_collection' in df.columns:
            df['belongs_to_collection'] = df['belongs_to_collection'].apply(
                lambda x: x['name'] if isinstance(x, dict) else None
            )
            
        # 2. Handle Lists of Dicts (Genres, Production, etc.)
        list_cols = ['genres', 'spoken_languages', 'production_countries', 'production_companies', 'credits']
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: "|".join([item['name'] for item in x]) if isinstance(x, list) else None
                )
        return df

    def enforce_types_and_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts data types and scales financial units."""
        df = df.copy()
        
        # Numeric conversion
        numeric_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Date conversion
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

        # Unit Conversion & Zero handling
        # We replace 0 with NaN first to avoid 'free' movies skewing stats
        target_cols = ['budget', 'revenue', 'runtime']
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        
        # Scale to Millions
        df['budget'] = df['budget'] / 1e6
        df['revenue'] = df['revenue'] / 1e6
        
        return df.rename(columns={'budget': 'budget_musd', 'revenue': 'revenue_musd'})

    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates, low-info rows, and unreleased content."""
        df = df.copy()
        
        initial_count = len(df)
        df.dropna(subset=['id', 'title'], inplace=True)
        df.drop_duplicates(subset=['id'], inplace=True)
        
        # Only keep 'Released' status if it exists, then drop the column
        if 'status' in df.columns:
            df = df[df['status'] == 'Released']
            df = df.drop(columns=['status'])
            
        # Quality threshold: Drop rows with less than 10 non-null values
        df = df.dropna(thresh=10)
        
        self.logger.info(f"Quality Filter: {initial_count} -> {len(df)} rows")
        return df

    def run_transformation(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Orchestrates the full transformation pipeline."""
        self.logger.info("Starting transformation pipeline...")
        
        # Drop irrelevant columns immediately
        cols_to_drop = ['adult', 'imdb_id', 'original_title', 'video', 'homepage']
        df = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns])

        # Execute Pipeline
        df = (df.pipe(self.flatten_json_columns)
                .pipe(self.enforce_types_and_units)
                .pipe(self.filter_quality))

        # Final Schema Enforcement
        target_order = [
            'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
            'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
            'production_countries', 'vote_count', 'vote_average', 'popularity', 'runtime',
            'overview', 'spoken_languages', 'poster_path'
        ]
        df = df.reindex(columns=target_order).reset_index(drop=True)
        
        self._save_to_csv(df)
        return df

    def _save_to_csv(self, df: pd.DataFrame):
        output_path = os.path.join(self.processed_data_dir, "movies_clean.csv")
        df.to_csv(output_path, index=False)
        self.logger.info(f"Cleaned data saved to {output_path}")