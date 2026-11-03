# ⚽ Player Injuries & Team Performance Dashboard

**FootLens Analytics | AI Research & Insights Team**
Mathematics for AI-II · Summative Assessment · Scenario 1

**Student:** Het Thakkar  |  **Student ID:** 2505566

## 🔗 Live Dashboard

**https://playersinjuriesimpactpy-67.streamlit.app**

---

## 1. Project Overview

Football clubs invest heavily in their squads, yet injuries to key players often decide the outcome of a season. This project analyses how much a team really suffers when a player is injured, using Premier League injury data from the 2019/20 to 2023/24 seasons.

The dashboard was built as an interactive Streamlit application for the FootLens Analytics research team. It turns a raw injury log into answers to five business questions:

1. Which injuries lead to the biggest drop in team performance?
2. What is the team's win/draw/loss record while a player is absent?
3. How do players perform after they recover?
4. Are there months or clubs with frequent injury clusters?
5. Which clubs suffer most because of injuries?

### Key findings

| Question | Finding |
|---|---|
| Most damaging injury (5+ cases) | **Knee surgery**: about 1.42 goals of goal difference lost per match (6 cases) |
| Common injuries with large impact | Muscle injury (0.77, 27 cases) and groin injury (0.58, 25 cases) |
| Team record while a player is absent | 581 wins, 374 draws, 548 losses (win rate about 39%) |
| Recovery | Average rating change of +0.03, so players return at roughly their previous level |
| Peak injury months | December (94), January (73), November (70) |
| Clubs most affected | Brentford (0.62) and Burnley (0.54) average goal-difference drop per match |
| Age effect | Almost none (correlation r = 0.03 between age and performance drop) |

---

## 2. Dataset

- **File:** `player_injuries_impact.csv`
- **Size:** about 656 raw records, 42 columns (654 records after cleaning, 224 players)
- **Contents:** player, club, position, age, season, FIFA rating, injury type, injury and return dates, and for the three matches before the injury, the matches missed, and the three matches after the injury: result, goal difference and player rating.

---

## 3. Data Cleaning & Feature Engineering

**Cleaning**
- Renamed unclear columns to a consistent `snake_case` format
- Converted text dates to datetime. Fixed dates with a missing space after the comma, impossible years (such as 0001 and 0202), and "Present" return dates
- Converted text ratings and goal differences to numbers (placeholders like `N.A.` became missing values)
- Standardised match results to Win / Draw / Loss
- Removed duplicates, rows without player/club/injury date, and negative injury durations
- Filled missing age with the median, and missing FIFA ratings with the club median

**New columns**
- `injury_days`: length of the injury in days, plus a severity band
- `avg_rating_before`, `avg_rating_after`, `rating_change`: player form around the injury
- `perf_drop_index`: average goal difference before the injury minus average goal difference in the matches missed (higher = bigger drop)
- `injury_month`, `position_group`, `missed_wins / draws / losses`

---

## 4. Dashboard Features

- Sidebar filters for season, club, position and age range
- KPI cards summarising records, players, days out, team drop index and rating change
- Auto-generated answers to the five business questions
- Five required visualisations:
  1. Bar chart: top 10 injuries by team performance drop
  2. Line chart: player rating timeline before and after an injury
  3. Heatmap: injury frequency by month and club
  4. Scatter plot: player age vs performance drop index (with trend line)
  5. Comeback leaderboard: players with the biggest rating improvement
- Bonus chart: match results while a player was absent, by club
- Data & Cleaning tab with a cleaning summary, summary statistics and a cleaned-data download

---

## 5. Integration Details

| Layer | Tool | Role |
|---|---|---|
| Data processing | pandas, numpy | Load, clean and engineer features from the CSV |
| Visualisation | Plotly | Interactive charts with a shared dark theme |
| Application | Streamlit | Filters, KPI cards, tabs and page layout |
| Version control | GitHub | Hosts the code, data and dependency list |
| Hosting | Streamlit Community Cloud | Builds and serves the app from the GitHub repo |

**How it fits together:** the app reads `player_injuries_impact.csv` from the same folder as the script, cleans it once, and caches the result so filters respond instantly. Every chart is built from the filtered data, so the sidebar updates the whole dashboard. When code is pushed to GitHub, Streamlit Cloud redeploys the app automatically.

---

## 6. Deployment Instructions

### Run locally

1. Clone the repository and open the folder:
   ```
   git clone https://github.com/Het-thakkar/IADAI102-2505566-het-thakkar.git
   cd IADAI102-2505566-het-thakkar
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run player_injuries_impact.py
   ```
4. Open `http://localhost:8501` in your browser.

### Deploy on Streamlit Community Cloud

1. Push `player_injuries_impact.py`, `requirements.txt` and `player_injuries_impact.csv` to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repository, branch `main`, and the main file `player_injuries_impact.py`.
4. Click **Deploy**. The app builds in a few minutes and gets a public link.

---

## 7. Repository Structure

```
├── player_injuries_impact.py    # Streamlit application
├── player_injuries_impact.csv   # Dataset
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 8. Screenshots

![Dashboard overview](screenshots/overview.png)

![Visual analysis](screenshots/visuals.png)

---

## 9. Tech Stack

Python · pandas · numpy · Plotly · Streamlit · GitHub · Streamlit Community Cloud
