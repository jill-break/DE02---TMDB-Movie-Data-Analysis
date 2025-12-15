import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

logger.info("Starting visualization.py script")

start_time = time.time()

# Load cleaned data
data_dir = os.path.join(os.path.dirname(__file__), "../data/processed")
input_path = os.path.join(data_dir, "movies_clean.csv")
df_movie = pd.read_csv(input_path)
logger.info(f"Loaded cleaned data from {input_path} with {len(df_movie)} rows")

df_movie['is_franchise'] = df_movie['belongs_to_collection'].notna()

# Ensure numeric types
df_movie['budget_musd'] = pd.to_numeric(
    df_movie['budget_musd'], errors='coerce')
df_movie['revenue_musd'] = pd.to_numeric(
    df_movie['revenue_musd'], errors='coerce')
df_movie['popularity'] = pd.to_numeric(df_movie['popularity'], errors='coerce')
df_movie['vote_average'] = pd.to_numeric(
    df_movie['vote_average'], errors='coerce')

# Ensure Date is datetime type and extract year
df_movie['release_date'] = pd.to_datetime(df_movie['release_date'])
df_movie['release_year'] = df_movie['release_date'].dt.year

# Calculate ROI if missing
if 'roi' not in df_movie.columns:
    df_movie['roi'] = df_movie['revenue_musd'] / df_movie['budget_musd']

# Explode Genres for the ROI Chart
df_plot = df_movie.copy()
df_plot['genre_list'] = df_plot['genres'].str.split('|')
df_exploded = df_plot.explode('genre_list')

logger.info("Prepared data for visualization")

# 1. Revenue vs. Budget Trends (Scatter Plot)
plt.figure(figsize=(10, 6))

# Plot Standalone
plt.scatter(
    df_movie['budget_musd'],
    df_movie['revenue_musd'],
    color='skyblue',
    alpha=0.7,
    s=100,
    edgecolors='grey'
)
plt.title('Revenue vs. Budget Trends', fontsize=16)
plt.xlabel('Budget (Million USD)', fontsize=12)
plt.ylabel('Revenue (Million USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(['Individual Movies'], loc='upper right')

plt.show()

logger.info("Generated Revenue vs. Budget scatter plot")

# 2. ROI Distribution by Genre (Bar Chart)
plt.figure(figsize=(12, 6))

# Calculate Median ROI per genre and sort
genre_roi = df_exploded.groupby('genre_list')['roi'].median().sort_values()

# Plot Horizontal Bar Chart
bars = plt.barh(genre_roi.index, genre_roi.values,
                color='lightgreen', edgecolor='black')

plt.title('Median ROI Distribution by Genre', fontsize=16)
plt.xlabel('Median ROI (x Times Budget)', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5, axis='x')

# Add Labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}x', va='center', fontsize=10, color='black')

plt.tight_layout()

plt.legend(['Median ROI per Genre'], loc='upper right')

plt.show()

logger.info("Generated ROI Distribution by Genre bar chart")

logger.info("Generated ROI Distribution by Genre bar chart")

# 3. Popularity vs. Rating (Scatter Plot)
plt.figure(figsize=(10, 6))

plt.scatter(
    df_movie['popularity'],
    df_movie['vote_average'],
    color='purple',
    alpha=0.6,
    edgecolors='black',
    s=80
)

plt.title('Relationship: Popularity vs. User Rating', fontsize=16)
plt.xlabel('Popularity Score', fontsize=12)
plt.ylabel('Vote Average (0-10)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(['Individual Movies'], loc='upper right')

plt.show()

logger.info("Generated Popularity vs. Rating scatter plot")

# 4. Yearly Trends in Box Office Performance (Line Chart)
df_movie["release_year"] = df_movie["release_date"].dt.year

# Group by year and calculate metrics
yearly_stats = df_movie.groupby("release_year").agg(
    {"revenue_musd": ["count", "mean"], "budget_musd": "mean", "roi": "mean"}
)

yearly_stats.columns = ["Movie Count",
                        "Mean Revenue", "Mean Budget", "Mean ROI"]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Yearly Box Office Performance Trends",
             fontsize=16, fontweight="bold")

# Movie Count per Year
axs[0, 0].plot(yearly_stats.index, yearly_stats["Movie Count"], marker="o")
axs[0, 0].set_title("Number of Movies Released per Year")
axs[0, 0].set_ylabel("Count")
axs[0, 0].legend(['Movie Count'], loc='upper left')

# Average Revenue per Year
axs[0, 1].plot(
    yearly_stats.index, yearly_stats["Mean Revenue"], marker="o", color="green"
)
axs[0, 1].set_title("Average Revenue per Year (M USD)")
axs[0, 1].set_ylabel("Revenue (M USD)")
axs[0, 1].legend(['Mean Revenue'], loc='upper left')

# Average Budget per Year
axs[1, 0].plot(
    yearly_stats.index, yearly_stats["Mean Budget"], marker="o", color="orange"
)
axs[1, 0].set_title("Average Budget per Year (M USD)")
axs[1, 0].set_ylabel("Budget (M USD)")
axs[1, 0].legend(['Mean Budget'], loc='upper left')

# Average ROI per Year
axs[1, 1].plot(
    yearly_stats.index, yearly_stats["Mean ROI"], marker="o", color="purple"
)
axs[1, 1].set_title("Average ROI per Year")
axs[1, 1].set_ylabel("ROI")
axs[1, 1].legend(['Mean ROI'], loc='upper left')

for ax in axs.flat:
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

logger.info("Generated Yearly Trends line chart")

# 5. Franchise vs. Standalone Success (Bar Chart)
franchise_stats = df_movie[df_movie["belongs_to_collection"].notna()]
standalone = df_movie[df_movie["belongs_to_collection"].isna()]

# calculate metrics
metrics = {
    "Revenue": [
        franchise_stats["revenue_musd"].mean(),
        standalone["revenue_musd"].mean(),
    ],
    "ROI": [franchise_stats["roi"].mean(), standalone["roi"].mean()],
    "Budget": [franchise_stats["budget_musd"].mean(), standalone["budget_musd"].mean()],
    "Rating": [franchise_stats["vote_average"].mean(), standalone["vote_average"].mean()],
}

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "Franchise vs Standalone Movie Performance", fontsize=14, fontweight="bold"
)

categories = ["Franchise", "Standalone"]
colors = ["#2E86AB", "#A23B72"]

# Revenue comparison
axs[0, 0].bar(categories, metrics["Revenue"], color=colors, edgecolor="black")
axs[0, 0].set_title("Average Revenue (M USD)")
axs[0, 0].set_ylabel("Million USD")
axs[0, 0].grid(True, axis="y", alpha=0.3)

# ROI comparison
axs[0, 1].bar(categories, metrics["ROI"], color=colors, edgecolor="black")
axs[0, 1].set_title("Average ROI")
axs[0, 1].set_ylabel("ROI Multiplier")
axs[0, 1].grid(True, axis="y", alpha=0.3)

# Budget comparison
axs[1, 0].bar(categories, metrics["Budget"], color=colors, edgecolor="black")
axs[1, 0].set_title("Average Budget (M USD)")
axs[1, 0].set_ylabel("Million USD")
axs[1, 0].grid(True, axis="y", alpha=0.3)

# Rating comparison
axs[1, 1].bar(categories, metrics["Rating"], color=colors, edgecolor="black")
axs[1, 1].set_title("Average Rating")
axs[1, 1].set_ylabel("Rating (out of 10)")
axs[1, 1].set_ylim(0, 10)
axs[1, 1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[0]),
           plt.Rectangle((0, 0), 1, 1, color=colors[1])]
fig.legend(handles, categories, loc='upper center',
           bbox_to_anchor=(0.5, 1.05), ncol=2)

plt.show()

logger.info("Generated Franchise vs. Standalone bar chart")

end_time = time.time()
logger.info(
    f"visualization.py completed successfully in {end_time - start_time:.2f} seconds")
