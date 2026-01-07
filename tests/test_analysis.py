import pytest
import pandas as pd
import numpy as np
import logging
from src.transformation.analysis import MovieAnalyzer

def test_franchise_analysis_single_category():
    """Ensure no 'Length Mismatch' error occurs when data has only one category."""
    # Fixed: "pd" and "np" now defined via imports
    df = pd.DataFrame({
        'id': [1, 2],
        'title': ['A', 'B'],
        'revenue_musd': [100, 200],
        'budget_musd': [50, 100],
        'popularity': [10.5, 12.0],        # Added
        'roi': [2.0, 2.0],                 # Added
        'vote_average': [7.5, 8.0],        # Added
        'belongs_to_collection': [np.nan, np.nan]
    })
    
    analyzer = MovieAnalyzer.__new__(MovieAnalyzer)
    analyzer.df = df
    analyzer.logger = logging.getLogger("Test")
    
    stats = analyzer.get_franchise_comparison()
    
    assert "Standalone" in stats.index

def test_roi_calculation_with_zero_budget():
    """Verify that ROI handles zero budget by returning NaN instead of inf."""
    # Fixed: "pd" now defined
    df_zero_budget = pd.DataFrame({
        'id': [1],
        'title': ['Zero Budget Movie'],
        'budget_musd': [0],
        'revenue_musd': [100]
    })
    
    analyzer = MovieAnalyzer.__new__(MovieAnalyzer)
    analyzer.df = df_zero_budget
    analyzer.logger = logging.getLogger("Test")
    
    # Trigger the internal KPI logic
    analyzer._calculate_base_kpis()
    
    # Fixed: "np" now defined
    assert np.isnan(analyzer.df.iloc[0]['roi'])