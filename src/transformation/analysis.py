import os
import logging
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Union

class MovieAnalyzer:
    """
    Expert-level analyzer that implements all Step 3 KPI and 
    Advanced Search requirements.
    """
    
    def __init__(self, cleaned_csv_path: str):
        """
        Initialize the MovieAnalyzer with cleaned data.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not os.path.exists(cleaned_csv_path):
            self.logger.error(f"Cleaned data missing: {cleaned_csv_path}")
            raise FileNotFoundError(f"Ensure process.py has run first.")
            
        self.df = pd.read_csv(cleaned_csv_path)
        self.logger.info(f"Loaded {len(self.df)} rows from cleaned CSV.")
        self._calculate_base_kpis()

    def _calculate_base_kpis(self) -> None:
        """Requirement 1: KPI Implementation (Profit & ROI)."""
        if 'profit_musd' not in self.df.columns:
            self.df['profit_musd'] = self.df['revenue_musd'] - self.df['budget_musd']
        
        if 'roi' not in self.df.columns:
            # ROI = Revenue / Budget
            self.df['roi'] = np.where(
                self.df['budget_musd'] > 0, 
                self.df['revenue_musd'] / self.df['budget_musd'], 
                np.nan
            )
        self.logger.info("KPIs verified.")

    def enrich_with_credits(self, raw_json_path: str) -> None:
        """Requirement 3: Merging for Advanced Filtering (Actor/Director)."""
        with open(raw_json_path, 'r') as f:
            raw_data = json.load(f)
        
        df_credits = pd.DataFrame(raw_data)[['id', 'credits']]
        self.df = pd.merge(self.df, df_credits, on='id', how='left')
        
        self.df['cast'] = self.df['credits'].apply(self._extract_cast)
        self.df['director'] = self.df['credits'].apply(self._extract_director)
        self.df.drop(columns=['credits'], inplace=True)
        self.logger.info("Enriched cleaned data with cast/director features.")

    @staticmethod
    def _extract_cast(credits_data: Union[dict, float, None]) -> str:
        if isinstance(credits_data, dict) and 'cast' in credits_data:
            return "|".join([x['name'] for x in credits_data['cast'][:5]])
        return ""

    @staticmethod
    def _extract_director(credits_data: Union[dict, float, None]) -> str:
        if isinstance(credits_data, dict) and 'crew' in credits_data:
            return "|".join([x['name'] for x in credits_data['crew'] if x.get('job') == 'Director'])
        return ""

    # --- REQUIREMENT 2: UDF FOR RANKING ---
    def rank_movies(self, 
                    criteria_col: str, 
                    ascending: bool = False, 
                    top_n: int = 5, 
                    mask: Optional[pd.Series] = None,
                    show_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """User-Defined Function (UDF) to streamline ranking operations."""
        target_df = self.df[mask] if mask is not None else self.df
        
        if show_cols is None:
            show_cols = ['title', criteria_col, 'genres', 'revenue_musd', 'profit_musd', 'roi', 'vote_average']
            
        ranked = target_df.sort_values(by=criteria_col, ascending=ascending).head(top_n)
        return ranked[[c for c in show_cols if c in ranked.columns]]

    # --- REQUIREMENT 4: FRANCHISE VS STANDALONE ---
    def get_franchise_comparison(self) -> pd.DataFrame:
        """Compare franchises vs standalone movies across multiple means."""
        self.df['is_franchise'] = self.df['belongs_to_collection'].notna()
        
        stats = self.df.groupby('is_franchise').agg({
            'revenue_musd': 'mean',
            'roi': 'median',
            'budget_musd': 'mean',
            'popularity': 'mean',
            'vote_average': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'movie_count'})
        
        label_map = {False: 'Standalone', True: 'Franchise'}
        stats.index = stats.index.map(label_map)
        return stats.round(2)

    # --- REQUIREMENT 5: SUCCESSFUL FRANCHISES ---
    def get_most_successful_franchises(self, top_n: int = 5) -> pd.DataFrame:
        """Rank franchises by total revenue and counts."""
        franchise_df = self.df[self.df['belongs_to_collection'].notna()]
        
        success = franchise_df.groupby('belongs_to_collection').agg({
            'title': 'count',
            'budget_musd': ['sum', 'mean'],
            'revenue_musd': ['sum', 'mean'],
            'vote_average': 'mean'
        })
        
        # Flatten MultiIndex columns
        success.columns = ['_'.join(col).strip() for col in success.columns.values]
        return success.sort_values(by='revenue_musd_sum', ascending=False).head(top_n).round(2)

    # --- REQUIREMENT 6: SUCCESSFUL DIRECTORS ---
    def get_most_successful_directors(self, top_n: int = 5) -> pd.DataFrame:
        """Rank directors by total revenue and movie count."""
        success = self.df[self.df['director'] != ""].groupby('director').agg({
            'title': 'count',
            'revenue_musd': 'sum',
            'vote_average': 'mean'
        }).rename(columns={'title': 'total_movies', 'revenue_musd': 'total_revenue'})
        
        return success.sort_values(by='total_revenue', ascending=False).head(top_n).round(2)