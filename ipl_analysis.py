# IPL / Sports Data Analysis - Task 2
# Paste into a Jupyter cell or save as ipl_analysis.py and run.
# Requires: pandas, numpy, matplotlib, seaborn, requests, fpdf (optional for PDF)
# Install missing packages (uncomment below if needed):
# !pip install pandas numpy matplotlib seaborn requests fpdf

import os
import io
import requests
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set(style="whitegrid", font_scale=1.05)

# ---------------------------
# 1) Download dataset (with a couple of public fallbacks)
# ---------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Candidate raw URLs (GitHub repos that host matches/deliveries)
candidates = [
    # repo: sanjay-zeliot (has "IPL Ball-by-Ball 2008-2020.csv" and "IPL Matches 2008-2020.csv")
    ("https://raw.githubusercontent.com/sanjay-zeliot/IPL-DataAnalysis/main/IPL%20Matches%202008-2020.csv",
     "data/matches.csv"),
    ("https://raw.githubusercontent.com/sanjay-zeliot/IPL-DataAnalysis/main/IPL%20Ball-by-Ball%202008-2020.csv",
     "data/deliveries.csv"),
    # other public repos (fallbacks)
    ("https://raw.githubusercontent.com/atwyburde/IPL-Data-Analysis/main/Data/IPL_Matches_2008_2021.csv",
     "data/matches_alt.csv"),
    ("https://raw.githubusercontent.com/atwyburde/IPL-Data-Analysis/main/Data/IPL_Ball_by_Ball_2008_2021.csv",
     "data/deliveries_alt.csv"),
]

def download_try(url, out_path):
    try:
        print(f"Attempting download: {url}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            with open(out_path, "wb") as f:
                f.write(r.content)
            print("Saved:", out_path)
            return True
        else:
            print("Failed to download or file too small:", url, "status:", r.status_code)
            return False
    except Exception as e:
        print("Error downloading", url, e)
        return False

# Try first pair (expected standard names)
if not os.path.exists("data/matches.csv") or not os.path.exists("data/deliveries.csv"):
    # Try primary pair first
    ok1 = download_try(candidates[0][0], candidates[0][1])
    ok2 = download_try(candidates[1][0], candidates[1][1])
    # If primary didn't work, try fallbacks
    if not (ok1 and ok2):
        print("Primary raw links didn't both succeed; trying alternative links...")
        download_try(candidates[2][0], candidates[2][1])
        download_try(candidates[3][0], candidates[3][1])

# After attempts, decide which files to load
def choose_file(pref, alt):
    if os.path.exists(pref):
        return pref
    elif os.path.exists(alt):
        return alt
    else:
        return None

matches_path = choose_file("data/matches.csv", "data/matches_alt.csv")
deliveries_path = choose_file("data/deliveries.csv", "data/deliveries_alt.csv")

if not matches_path or not deliveries_path:
    raise FileNotFoundError("Could not download matches/deliveries CSVs. "
                            "If running locally, please download from Kaggle (nowke9/ipldata) or from a public GitHub IPL dataset and place files in ./data/")

print("Using matches:", matches_path)
print("Using deliveries:", deliveries_path)

# ---------------------------
# 2) Load data
# ---------------------------
matches = pd.read_csv(matches_path, low_memory=False)
deliveries = pd.read_csv(deliveries_path, low_memory=False)
print("Matches rows:", len(matches), "Deliveries rows:", len(deliveries))

# Normalize column names: lower and replace spaces
matches.columns = [c.strip().lower().replace(" ", "_") for c in matches.columns]
deliveries.columns = [c.strip().lower().replace(" ", "_") for c in deliveries.columns]

# Parse dates if available
for df in [matches]:
    if "date" in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            pass

# ---------------------------
# 3) Quick cleaning & harmonization (depends on dataset variant)
# ---------------------------
# Common expected columns:
# matches: id (or match_id), season/year, winner, team1, team2, city, venue, toss_winner, toss_decision, result_margin
# deliveries: match_id (id), inning, over, ball, batsman, non_striker, bowler, batsman_runs, total_runs, extra_runs, player_dismissed

# attempt to find match id column name
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

match_id_col = find_col(matches, ["id","match_id","matchno"])
delivery_match_id = find_col(deliveries, ["match_id","id","matchid"])

print("match_id_col:", match_id_col, "delivery_match_id:", delivery_match_id)

# standardize names
if match_id_col and delivery_match_id:
    matches = matches.rename(columns={match_id_col: "match_id"})
    deliveries = deliveries.rename(columns={delivery_match_id: "match_id"})

# season/year column
season_col = find_col(matches, ["season","year"])
if season_col and season_col != "season":
    matches = matches.rename(columns={season_col: "season"})

# ---------------------------
# 4) Compute KPIs
# ---------------------------

# Top run scorers
if "batsman_runs" in deliveries.columns:
    runs_by_batsman = deliveries.groupby("batsman")["batsman_runs"].sum().sort_values(ascending=False)
else:
    possible = find_col(deliveries, ["batsman_run", "runs", "batsman_runs"])
    deliveries = deliveries.rename(columns={possible: "batsman_runs"})
    runs_by_batsman = deliveries.groupby("batsman")["batsman_runs"].sum().sort_values(ascending=False)

top_batsmen = runs_by_batsman.head(20).reset_index()
top_batsmen.to_csv("outputs/top_batsmen.csv", index=False)

# Strike rate
if "wide_runs" in deliveries.columns:
    valid_balls = deliveries[deliveries["wide_runs"].fillna(0) == 0]
else:
    valid_balls = deliveries.copy()

balls_df = valid_balls.groupby("batsman").size().reset_index(name="balls")
runs_df = deliveries.groupby("batsman")["batsman_runs"].sum().reset_index().rename(columns={"batsman_runs": "runs"})
sr_df = pd.merge(runs_df, balls_df, on="batsman", how="left")
sr_df = sr_df[sr_df["balls"] > 0].copy()
sr_df["strike_rate"] = (sr_df["runs"] / sr_df["balls"]) * 100
sr_df = sr_df.sort_values("runs", ascending=False)
sr_df.head(20).to_csv("outputs/top_batsmen_strike_rates.csv", index=False)

# ✅ Team win rates (with safe fallback)
team_stats = pd.DataFrame()
if all(col in matches.columns for col in ["team1", "team2", "winner"]):
    total_matches_by_team = pd.concat([matches["team1"], matches["team2"]]).value_counts().rename("matches_played")
    wins = matches["winner"].value_counts().rename("wins")
    team_stats = pd.concat([total_matches_by_team, wins], axis=1).fillna(0)
    team_stats["win_rate_pct"] = (team_stats["wins"] / team_stats["matches_played"]) * 100
    team_stats = team_stats.sort_values("wins", ascending=False)
    team_stats.to_csv("outputs/team_win_rates.csv")
else:
    print("⚠️ Could not compute team win rates — 'team1', 'team2', or 'winner' column missing.")


# ---------------------------
# 5) Visualizations
# ---------------------------

# Helper plotting functions
def savefig(fig, name, dpi=150):
    out = os.path.join("outputs", name)
    fig.savefig(out, bbox_inches='tight', dpi=dpi)
    print("Saved figure:", out)

# Top 10 run-scorers overall (bar)
fig, ax = plt.subplots(figsize=(10,6))
top20 = runs_by_batsman.head(10)
sns.barplot(x=top20.values, y=top20.index, ax=ax)
ax.set_title("Top 10 Run Scorers (Overall)")
ax.set_xlabel("Runs")
ax.set_ylabel("Batsman")
savefig(fig, "top10_runs_overall.png")
plt.close(fig)

# Top 10 strike rates (min balls threshold)
min_balls = 200
sr_filtered = sr_df[sr_df["balls"] >= min_balls].sort_values("strike_rate", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x="strike_rate", y="batsman", data=sr_filtered, ax=ax)
ax.set_title(f"Top Strike Rates (>= {min_balls} balls faced)")
ax.set_xlabel("Strike Rate")
ax.set_ylabel("Batsman")
savefig(fig, "top_strike_rates.png")
plt.close(fig)

# Team win rates (bar)
if 'win_rate_pct' in team_stats.columns:
    fig, ax = plt.subplots(figsize=(10,6))
    ts = team_stats.sort_values("win_rate_pct", ascending=False).reset_index()
    sns.barplot(y=ts['index'], x=ts['win_rate_pct'], ax=ax)
    ax.set_title("Team Win Rate (%)")
    ax.set_xlabel("Win rate (%)")
    ax.set_ylabel("Team")
    savefig(fig, "team_win_rates.png")
    plt.close(fig)

# Per-season top scorer trend (example: top 5 across seasons)
if "season" in deliveries.columns:
    # pivot: seasons x batsmen runs for top batsmen overall
    top_players = runs_by_batsman.head(10).index.tolist()
    per_season = deliveries[deliveries['batsman'].isin(top_players)].groupby(['season','batsman'])['batsman_runs'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11,6))
    for player in top_players:
        d = per_season[per_season['batsman']==player]
        ax.plot(d['season'], d['batsman_runs'], marker='o', label=player)
    ax.set_title("Season-wise runs for top players (sample)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Runs")
    ax.legend(loc='best', fontsize='small')
    savefig(fig, "per_season_top_players.png")
    plt.close(fig)

# Create a one-page visual summary (composite)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
fig = plt.figure(figsize=(11,14))
canvas = FigureCanvas(fig)
gs = fig.add_gridspec(3,1)

# subplot 1: top runs
ax1 = fig.add_subplot(gs[0,0])
top20 = runs_by_batsman.head(8)
sns.barplot(x=top20.values, y=top20.index, ax=ax1)
ax1.set_title("Top 8 Run Scorers (Overall)")

# subplot 2: strike rates
ax2 = fig.add_subplot(gs[1,0])
sr_plot = sr_df[sr_df["balls"]>=min_balls].sort_values("strike_rate", ascending=False).head(8)
sns.barplot(x="strike_rate", y="batsman", data=sr_plot, ax=ax2)
ax2.set_title(f"Top Strike Rates (min {min_balls} balls)")

# subplot 3: team win rates (top 8)
ax3 = fig.add_subplot(gs[2,0])
if 'win_rate_pct' in team_stats.columns:
    ts8 = team_stats.sort_values("win_rate_pct", ascending=False).head(8).reset_index()
    sns.barplot(x=ts8['win_rate_pct'], y=ts8['index'], ax=ax3)
    ax3.set_xlabel("Win rate (%)")
    ax3.set_title("Top Team Win Rates")
else:
    ax3.text(0.1,0.5,"No team win data available", fontsize=12)

fig.suptitle("IPL - Visual Summary", fontsize=16)
savefig(fig, "visual_summary_onepage.png")
plt.close(fig)

# ---------------------------
# 6) Short text insights (automatically generated)
# ---------------------------
insights = []
# Top batsman overall
top1 = runs_by_batsman.index[0]
top1_runs = runs_by_batsman.iloc[0]
insights.append(f"Top run scorer overall: {top1} with {int(top1_runs)} runs (based on deliveries.csv aggregation).")

# Team with most wins
if 'wins' in team_stats.columns:
    top_team = team_stats['wins'].idxmax()
    insights.append(f"Team with most wins (in dataset): {top_team} with {int(team_stats.loc[top_team,'wins'])} recorded wins.")

# Strike rate highlight
best_sr = sr_df.sort_values("strike_rate", ascending=False).iloc[0]
insights.append(f"Highest strike rate (min balls threshold applied): {best_sr['batsman']} - SR {best_sr['strike_rate']:.2f} (runs: {int(best_sr['runs'])}, balls: {int(best_sr['balls'])}).")

with open("outputs/summary_insights.txt", "w", encoding="utf-8") as f:
    f.write("IPL Analysis - Short Insights\n")
    f.write("=============================\n\n")
    for s in insights:
        f.write("- " + s + "\n")
print("Wrote insights to outputs/summary_insights.txt")

# ---------------------------
# 7) Prepare README snippet for GitHub submission
# ---------------------------
readme = f"""
# Syntecxhub_IPL_Analysis (Task 2 - IPL / Sports Data Analysis)

This repository contains an end-to-end analysis for the Syntecxhub Task 2 (IPL / Sports data analysis).
Files generated by the notebook/script are in `outputs/`.

## What this project does
- Downloads public IPL datasets (matches & deliveries).
- Computes KPIs:
  - Top run scorers overall & per-season
  - Strike rates (with ball-face threshold)
  - Team win rates
- Produces visualizations:
  - Top scorers, strike rates, per-season comparisons, team win rates
- Exports:
  - `outputs/visual_summary_onepage.png` (one-page visual summary)
  - `outputs/*.png` (individual charts)
  - `outputs/summary_insights.txt` (short insights)

## How to run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn requests fpdf`
2. Run the notebook `ipl_analysis.ipynb` or the script `ipl_analysis.py`.
3. All outputs will be saved under `outputs/`.

## Notes
Dataset sources (public mirrors used): GitHub repositories that host IPL matches/deliveries CSV files (mirrors of the Kaggle "IPL dataset"). See `data/` for downloaded CSVs.
"""

with open("outputs/README_FOR_GITHUB.txt", "w", encoding="utf-8") as f:
    f.write(readme)

print("Project outputs saved in ./outputs/ .")
