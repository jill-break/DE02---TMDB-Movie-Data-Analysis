import pytest
import pandas as pd
import numpy as np
from src.transformation.process import MovieTransformer

@pytest.fixture
def sample_raw_data():
    """Provides a small mock dataframe mimicking TMDB API response."""
    return pd.DataFrame([{
        'id': 100,
        'title': 'Test Movie',
        'budget': 1000000,
        'revenue': 2000000,
        'status': 'Released',
        'genres': [{'name': 'Action'}, {'name': 'Sci-Fi'}],
        'belongs_to_collection': {'name': 'Test Collection'}
    }])

def test_financial_conversion(sample_raw_data):
    """Verify that budget/revenue are converted to Millions (MUSD)."""
    transformer = MovieTransformer()
    
    # FIX: Match the parameter name 'df' to satisfy Pylance
    transformer._save_to_csv = lambda df: None 
    
    df_clean = transformer.run_transformation(sample_raw_data)
    
    assert df_clean.iloc[0]['budget_musd'] == 1.0
    assert df_clean.iloc[0]['revenue_musd'] == 2.0

def test_json_flattening(sample_raw_data):
    """Verify that nested lists/dicts are converted to pipe-separated strings."""
    transformer = MovieTransformer()
    
    # FIX: Match the parameter name 'df' to satisfy Pylance
    transformer._save_to_csv = lambda df: None
    
    df_clean = transformer.run_transformation(sample_raw_data)
    
    assert df_clean.iloc[0]['genres'] == "Action|Sci-Fi"
    assert df_clean.iloc[0]['belongs_to_collection'] == "Test Collection"