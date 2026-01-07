import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

class MovieVisualizer:
    """A production-grade visualizer for movie performance data."""
    
    def __init__(self, df: pd.DataFrame):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df = df.copy()
        self._prepare_data()
        
        # Style configuration
        plt.style.use('ggplot') 
        self.output_dir = os.path.join(os.path.dirname(__file__), "../reports/figures")
        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_data(self):
        """Ensures all necessary plotting columns exist and are correctly typed."""
        # Convert to datetime and extract year if not present
        if 'release_date' in self.df.columns:
            self.df['release_date'] = pd.to_datetime(self.df['release_date'])
            self.df['release_year'] = self.df['release_date'].dt.year
            
        # Ensure is_franchise flag exists
        if 'belongs_to_collection' in self.df.columns:
            self.df['is_franchise'] = self.df['belongs_to_collection'].notna()
        
        self.logger.info("Visualizer data preparation complete.")

    def plot_revenue_vs_budget(self):
        """Generates a scatter plot showing financial correlations."""
        plt.figure(figsize=(10, 6))
        
        # Color code by franchise status for extra insight
        for label, group in self.df.groupby('is_franchise'):
            name = "Franchise" if label else "Standalone"
            plt.scatter(group['budget_musd'], group['revenue_musd'], 
                        label=name, alpha=0.6, s=100, edgecolors='white')

        plt.title('Financial Performance: Revenue vs. Budget')
        plt.xlabel('Budget (M USD)')
        plt.ylabel('Revenue (M USD)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_genre_roi(self):
        """Generates a horizontal bar chart of Median ROI per Genre."""
        # Explode genres safely
        df_exploded = self.df.assign(genre_list=self.df['genres'].str.split('|')).explode('genre_list')
        genre_roi = df_exploded.groupby('genre_list')['roi'].median().sort_values()

        if genre_roi.empty:
            self.logger.warning("No genre data available for ROI plot.")
            return

        plt.figure(figsize=(12, 8))
        bars = plt.barh(genre_roi.index, genre_roi.values, color='teal', edgecolor='black')
        
        # Label bars with the multiplier
        for bar in bars:
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                     f"{bar.get_width():.2f}x", va='center')

        plt.title('Median Return on Investment (ROI) by Genre')
        plt.xlabel('ROI Multiplier')
        plt.tight_layout()
        plt.show()

    def plot_yearly_trends(self):
        """Generates a multi-panel line chart for historical trends."""
        yearly = self.df.groupby('release_year').agg({
            'revenue_musd': 'mean',
            'budget_musd': 'mean',
            'roi': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'movie_count'})

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = [
            ('movie_count', 'Movies Released', 'blue'),
            ('revenue_musd', 'Avg Revenue (M USD)', 'green'),
            ('budget_musd', 'Avg Budget (M USD)', 'orange'),
            ('roi', 'Avg ROI', 'purple')
        ]

        for i, (col, title, color) in enumerate(metrics):
            ax = axs[i//2, i%2]
            ax.plot(yearly.index, yearly[col], marker='o', color=color, linewidth=2)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Yearly Box Office Performance Trends', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_franchise_comparison(self):
        """Compares Franchise vs Standalone across 4 key metrics."""
        # Defensive check: do we have both categories?
        if self.df['is_franchise'].nunique() < 2:
            self.logger.warning("Comparison plot skipped: Only one category (Franchise/Standalone) found.")
            return

        metrics = {
            "Revenue": "revenue_musd",
            "ROI": "roi",
            "Budget": "budget_musd",
            "Rating": "vote_average"
        }

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        categories = ['Standalone', 'Franchise']
        colors = ['#A23B72', '#2E86AB']

        for i, (title, col) in enumerate(metrics.items()):
            ax = axs[i//2, i%2]
            means = self.df.groupby('is_franchise')[col].mean()
            ax.bar(categories, means, color=colors, edgecolor='black')
            ax.set_title(f"Avg {title}")
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle("Franchise vs. Standalone Performance", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()