import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

class MovieVisualizer:
    """A production-grade visualizer that exports high-resolution PNG reports."""
    
    def __init__(self, df: pd.DataFrame):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df = df.copy()
        self._prepare_data()
        
        # Style configuration
        plt.style.use('ggplot') 
        # Define and create the export directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "../reports/figures")
        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_data(self):
        """Ensures all necessary plotting columns exist and are correctly typed."""
        if 'release_date' in self.df.columns:
            self.df['release_date'] = pd.to_datetime(self.df['release_date'])
            self.df['release_year'] = self.df['release_date'].dt.year
            
        if 'belongs_to_collection' in self.df.columns:
            self.df['is_franchise'] = self.df['belongs_to_collection'].notna()
        
        self.logger.info("Visualizer data preparation complete.")

    def _save_figure(self, filename: str):
        """Helper method to export plots with high-resolution settings."""
        save_path = os.path.join(self.output_dir, filename)
        # Use bbox_inches='tight' to ensure legends aren't cut off
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Figure exported to: {save_path}")

    def plot_revenue_vs_budget(self):
        """Scatter plot with legend and automated PNG export."""
        plt.figure(figsize=(10, 6))
        for label, group in self.df.groupby('is_franchise'):
            name = "Franchise" if label else "Standalone"
            plt.scatter(group['budget_musd'], group['revenue_musd'], 
                        label=f"{name} Movies", alpha=0.6, s=100, edgecolors='white')

        plt.title('Financial Performance: Revenue vs. Budget', fontsize=14)
        plt.xlabel('Budget (M USD)')
        plt.ylabel('Revenue (M USD)')
        plt.legend(title="Movie Category", loc='best', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        self._save_figure("revenue_vs_budget.png")
        plt.show()

    def plot_genre_roi(self):
        """Horizontal bar chart with ROI annotations and export."""
        df_exploded = self.df.assign(genre_list=self.df['genres'].str.split('|')).explode('genre_list')
        genre_roi = df_exploded.groupby('genre_list')['roi'].median().sort_values()

        if genre_roi.empty:
            self.logger.warning("No genre data available for ROI plot.")
            return

        plt.figure(figsize=(12, 8))
        bars = plt.barh(genre_roi.index, genre_roi.values, color='teal', edgecolor='black', label='Median ROI')
        
        for bar in bars:
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                     f"{bar.get_width():.2f}x", va='center', fontweight='bold')

        plt.title('Median Return on Investment (ROI) by Genre', fontsize=14)
        plt.xlabel('ROI Multiplier')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        self._save_figure("genre_roi_distribution.png")
        plt.show()

    def plot_popularity_vs_rating(self):
        """
        Generates a scatter plot to analyze the relationship 
        between movie popularity and user ratings.
        """
        plt.figure(figsize=(10, 6))

        plt.scatter(
            self.df['popularity'],
            self.df['vote_average'],
            color='purple',
            alpha=0.6,
            edgecolors='black',
            s=80,
            label='Individual Movies'
        )

        plt.title('Relationship: Popularity vs. User Rating', fontsize=16)
        plt.xlabel('Popularity Score (TMDB Metric)', fontsize=12)
        plt.ylabel('Vote Average (0-10 Scale)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Legend Annotation
        plt.legend(loc='upper right', frameon=True, shadow=True)

        # Automated High-Res Export
        self._save_figure("popularity_vs_rating.png")
        plt.show()

    def plot_yearly_trends(self):
        """Multi-panel line chart with export functionality."""
        yearly = self.df.groupby('release_year').agg({
            'revenue_musd': 'mean',
            'budget_musd': 'mean',
            'roi': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'movie_count'})

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [
            ('movie_count', 'Movies Released', 'blue', 'Count'),
            ('revenue_musd', 'Avg Revenue', 'green', 'M USD'),
            ('budget_musd', 'Avg Budget', 'orange', 'M USD'),
            ('roi', 'Avg ROI', 'purple', 'Multiplier')
        ]

        for i, (col, title, color, unit) in enumerate(metrics):
            ax = axs[i//2, i%2]
            ax.plot(yearly.index, yearly[col], marker='o', color=color, linewidth=2, label=f"{title} ({unit})")
            ax.set_title(title)
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Yearly Box Office Performance Trends', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        self._save_figure("yearly_performance_trends.png")
        plt.show()

    def plot_franchise_comparison(self):
        """Comparison bars with global legend and export."""
        if self.df['is_franchise'].nunique() < 2:
            self.logger.warning("Comparison plot skipped: Insufficient categories.")
            return

        metrics = {"Revenue": "revenue_musd", "ROI": "roi", "Budget": "budget_musd", "Rating": "vote_average"}
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        categories, colors = ['Standalone', 'Franchise'], ['#A23B72', '#2E86AB']

        for i, (title, col) in enumerate(metrics.items()):
            ax = axs[i//2, i%2]
            means = self.df.groupby('is_franchise')[col].mean()
            ax.bar(categories, means, color=colors, edgecolor='black')
            ax.set_title(f"Avg {title}")

        handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
        fig.legend(handles, categories, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, title="Comparison Groups")
        plt.suptitle("Franchise vs. Standalone Metrics", fontsize=16, y=1.05)
        plt.tight_layout()
        
        self._save_figure("franchise_vs_standalone.png")
        plt.show()