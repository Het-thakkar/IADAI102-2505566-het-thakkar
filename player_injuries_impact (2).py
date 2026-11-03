"""
FootLens Analytics - Player Injuries & Team Performance Dashboard
Mathematics for AI-II | Summative Assessment (Scenario 1)

Run locally:   streamlit run app.py
Data file:     player_injuries_impact.csv  (same folder as this file)
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FootLens | Injury Impact Dashboard",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#052e1a 0%,#0b3b2a 45%,#0b1220 100%);
  border:1px solid rgba(34,197,94,.35);border-radius:18px;padding:26px 32px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem;color:#fff;line-height:1.2}
.hero p{margin:6px 0 0;color:#9fb3c8}
.chip{display:inline-block;background:rgba(34,197,94,.15);color:#4ade80;
  border:1px solid rgba(34,197,94,.4);border-radius:999px;padding:3px 12px;
  font-size:.78rem;margin:12px 8px 0 0}
[data-testid="stMetric"]{background:linear-gradient(145deg,#111a2e,#0d1526);
  border:1px solid #1f2b45;border-radius:14px;padding:14px 18px;
  box-shadow:0 4px 14px rgba(0,0,0,.35)}
[data-testid="stMetricValue"]{color:#4ade80;font-weight:700}
[data-testid="stMetricLabel"]{color:#9fb3c8}
.insight{background:#111a2e;border:1px solid #1f2b45;border-left:4px solid #22c55e;
  border-radius:10px;padding:14px 18px;margin-bottom:12px}
.insight .q{color:#9fb3c8;font-size:.82rem;text-transform:uppercase;letter-spacing:.05em}
.insight .a{font-size:1.05rem;margin-top:4px}
.insight b{color:#4ade80}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background:#111a2e;border-radius:10px 10px 0 0;padding:8px 18px}
h2,h3{letter-spacing:-.01em}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "player_injuries_impact.csv"
GREEN, AMBER, RED, BLUE = "#22c55e", "#facc15", "#ef4444", "#38bdf8"
POS_COLORS = {"Goalkeeper": AMBER, "Defender": BLUE, "Midfielder": GREEN, "Forward": RED}
HEAT_SCALE = ["#0b1220", "#166534", "#facc15", "#ef4444"]
DROP_SCALE = ["#facc15", "#f97316", "#ef4444"]


def style_fig(fig, height=None):
    """One consistent dark look for every chart."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color="#dbe4f0"),
        margin=dict(l=10, r=10, t=30, b=10), legend_title_text="",
        hoverlabel=dict(bgcolor="#111a2e", font_size=13),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def insight(question: str, answer_html: str):
    """Styled answer card for a business question."""
    st.markdown(f'<div class="insight"><div class="q">{question}</div>'
                f'<div class="a">{answer_html}</div></div>', unsafe_allow_html=True)


def position_group(pos) -> str:
    p = str(pos).lower()
    if "keeper" in p:
        return "Goalkeeper"
    if "back" in p or "defend" in p:
        return "Defender"
    if "midfield" in p:
        return "Midfielder"
    return "Forward"
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RESULT_MAP = {"win": "Win", "w": "Win", "draw": "Draw", "d": "Draw",
              "lose": "Loss", "loss": "Loss", "l": "Loss", "lost": "Loss"}
POINTS_MAP = {"Win": 3, "Draw": 1, "Loss": 0}


# ----------------------------------------------------------------------------
# STEP 2: DATA LOADING & CLEANING
# ----------------------------------------------------------------------------
def to_snake(name: str) -> str:
    """Rename unclear column names -> clean snake_case."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()


def match_no(col: str) -> int:
    """Pull the match number (1, 2, 3...) out of a column name for sorting."""
    m = re.search(r"match(\d+)", col)
    return int(m.group(1)) if m else 0


def find_cols(df: pd.DataFrame, *patterns: str) -> list:
    """Find columns whose names match ALL regex patterns, sorted by match number."""
    cols = [c for c in df.columns if all(re.search(p, c, re.I) for p in patterns)]
    return sorted(cols, key=match_no)


def parse_dates(series: pd.Series) -> pd.Series:
    """Convert strings like 'Nov 9, 2019' or 'Oct 15,2022' to datetime.

    Fixes found in the raw data: missing space after the comma, 'Present'
    (player still injured) and impossible years such as 0001 / 0202.
    Anything unreadable becomes NaT instead of crashing.
    """
    s = series.astype(str).str.strip().str.replace(r",(?=\S)", ", ", regex=True)
    out = pd.to_datetime(s, format="%b %d, %Y", errors="coerce")
    missing = out.isna()
    if missing.any():  # fall back to flexible parsing for odd formats
        try:
            out[missing] = pd.to_datetime(s[missing], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            out[missing] = pd.to_datetime(s[missing], errors="coerce")
    out[out < pd.Timestamp("2000-01-01")] = pd.NaT  # impossible years
    return out


def mean_of(df: pd.DataFrame, cols: list) -> pd.Series:
    """Row-wise mean of several columns; NaN if the list is empty."""
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1, skipna=True)


@st.cache_data(show_spinner="Cleaning data...")
def load_and_clean(source) -> tuple:
    """Load the CSV, clean it, and engineer the new columns.

    Returns (clean_df, info_dict) where info_dict records what the cleaning did.
    """
    raw = pd.read_csv(source)
    info = {"raw_rows": len(raw), "raw_cols": raw.shape[1]}
    df = raw.copy()

    # 1. readable column names
    df.columns = [to_snake(c) for c in df.columns]

    # 2. strip whitespace in text columns, turn placeholders into NaN
    placeholders = ["n.a.", "n/a", "na", "nan", "none", "-", "", "null"]
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].str.lower().isin(placeholders), c] = np.nan

    # 3. remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    info["duplicates_removed"] = before - len(df)

    # 4. dates -> datetime
    df["date_of_injury"] = parse_dates(df["date_of_injury"])
    df["date_of_return"] = parse_dates(df["date_of_return"])
    info["return_date_missing"] = int(df["date_of_return"].isna().sum())

    # 5. numeric columns stored as text -> numbers
    for c in ["age", "fifa_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rating_cols = find_cols(df, r"rating$", r"(before|after)")
    gd_cols = find_cols(df, r"_gd$")
    for c in rating_cols + gd_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6. normalise match results to Win / Draw / Loss
    result_cols = find_cols(df, r"result$")
    for c in result_cols:
        df[c] = df[c].str.lower().map(RESULT_MAP)

    # 7. drop rows we cannot analyse (no player / team / injury date)
    before = len(df)
    df = df.dropna(subset=["name", "team_name", "date_of_injury"])
    info["rows_missing_key_fields"] = before - len(df)

    # 8. missing values: age -> median, fifa rating -> median of the team
    info["age_filled"] = int(df["age"].isna().sum())
    df["age"] = df["age"].fillna(df["age"].median())
    info["fifa_filled"] = int(df["fifa_rating"].isna().sum())
    df["fifa_rating"] = df["fifa_rating"].fillna(
        df.groupby("team_name")["fifa_rating"].transform("median")
    ).fillna(df["fifa_rating"].median())

    # 9. injury duration (days); remove impossible negatives
    df["injury_days"] = (df["date_of_return"] - df["date_of_injury"]).dt.days
    before = len(df)
    df = df[(df["injury_days"].isna()) | (df["injury_days"] >= 0)]
    info["negative_duration_removed"] = before - len(df)

    # 10. tidy categories
    df["injury"] = df["injury"].fillna("Unknown").str.title()
    df["position"] = df["position"].fillna("Unknown")
    df["position_group"] = df["position"].map(position_group)
    df["injury_month"] = df["date_of_injury"].dt.month.map(
        lambda m: MONTH_ORDER[int(m) - 1] if pd.notna(m) else np.nan
    )
    df["severity"] = pd.cut(
        df["injury_days"], bins=[-1, 14, 45, 90, 10_000],
        labels=["Minor (<=14d)", "Moderate (15-45d)",
                "Serious (46-90d)", "Severe (90d+)"],
    )

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------
    before_rating = find_cols(df, r"before", r"rating$")
    after_rating = find_cols(df, r"after", r"rating$")
    before_gd = find_cols(df, r"before", r"_gd$")
    missed_gd = find_cols(df, r"missed", r"_gd$")
    after_gd = find_cols(df, r"after", r"_gd$")

    df["avg_rating_before"] = mean_of(df, before_rating)
    df["avg_rating_after"] = mean_of(df, after_rating)
    df["rating_change"] = df["avg_rating_after"] - df["avg_rating_before"]

    df["avg_gd_before"] = mean_of(df, before_gd)
    df["avg_gd_missed"] = mean_of(df, missed_gd)
    df["avg_gd_after"] = mean_of(df, after_gd)

    # Team performance drop index: how much worse the goal difference is while
    # the player is absent compared with the matches before the injury.
    # Positive value = team did worse without the player.
    df["perf_drop_index"] = df["avg_gd_before"] - df["avg_gd_missed"]

    # Points per match (Win=3, Draw=1, Loss=0) before / during absence
    def points(prefix_pattern):
        cols = find_cols(df, prefix_pattern, r"result$")
        if not cols:
            return pd.Series(np.nan, index=df.index)
        return df[cols].apply(lambda s: s.map(POINTS_MAP)).mean(axis=1, skipna=True)

    df["ppm_before"] = points(r"before")
    df["ppm_missed"] = points(r"missed")

    # Count of wins/draws/losses while the player was absent
    missed_res = find_cols(df, r"missed", r"result$")
    for label, col in [("Win", "missed_wins"), ("Draw", "missed_draws"),
                       ("Loss", "missed_losses")]:
        df[col] = (df[missed_res] == label).sum(axis=1) if missed_res else 0

    info["clean_rows"] = len(df)
    info["cols_found"] = {
        "before_rating": before_rating, "after_rating": after_rating,
        "before_gd": before_gd, "missed_gd": missed_gd, "after_gd": after_gd,
        "missed_result": missed_res,
    }
    return df.reset_index(drop=True), info


# ----------------------------------------------------------------------------
# LOAD DATA (local file, otherwise let the user upload it)
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>⚽ Player Injuries &amp; Team Performance</h1>
  <p>FootLens Analytics &nbsp;|&nbsp; How much does a team really suffer when a key player is out?</p>
  <span class="chip">Premier League</span><span class="chip">2019/20 – 2023/24</span>
  <span class="chip">Injury → team impact</span>
</div>
""", unsafe_allow_html=True)

def find_data_file():
    """Look for the CSV next to this script, then in the current folder.
    Accepts slightly different names, e.g. 'player_injuries_impact (1).csv'."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        exact = os.path.join(folder, DATA_FILE)
        if os.path.exists(exact):
            return exact
        hits = sorted(glob.glob(os.path.join(folder, "*injur*.csv")))
        if hits:
            return hits[0]
    return None


source = find_data_file()
if source is None:
    source = st.file_uploader("Upload player_injuries_impact.csv", type="csv")
    if source is None:
        st.info("Place `player_injuries_impact.csv` next to app.py, or upload it above.")
        st.stop()

try:
    df, info = load_and_clean(source)
except Exception as err:  # keeps the deployed app from showing a raw traceback
    st.error(f"Could not read the dataset: {err}")
    st.stop()

if df.empty:
    st.warning("The dataset has no usable rows after cleaning.")
    st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
seasons = sorted(df["season"].dropna().unique())
sel_seasons = st.sidebar.multiselect("Season", seasons, default=seasons)
teams = sorted(df["team_name"].dropna().unique())
sel_teams = st.sidebar.multiselect("Club", teams, default=teams)
positions = sorted(df["position"].dropna().unique())
sel_pos = st.sidebar.multiselect("Position", positions, default=positions)
age_min, age_max = int(df["age"].min()), int(df["age"].max())
sel_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

f = df[
    df["season"].isin(sel_seasons)
    & df["team_name"].isin(sel_teams)
    & df["position"].isin(sel_pos)
    & df["age"].between(*sel_age)
]
if f.empty:
    st.warning("No records match these filters. Widen the filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# KPI ROW
# ----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Injury records", f"{len(f):,}")
k2.metric("Players", f"{f['name'].nunique():,}")
k3.metric("Avg. injury length", f"{f['injury_days'].mean():.0f} days")
k4.metric("Avg. team drop index", f"{f['perf_drop_index'].mean():.2f}")
k5.metric("Avg. rating change", f"{f['rating_change'].mean():+.2f}")

tab_over, tab_charts, tab_data = st.tabs(
    ["📌 Business Questions", "📊 Visual Analysis", "🧹 Data & Cleaning"]
)

# ----------------------------------------------------------------------------
# STEP 1 / 3: BUSINESS QUESTIONS + EDA
# ----------------------------------------------------------------------------
with tab_over:
    st.subheader("Five business questions, answered")

    # Q1: only injury types with 5+ cases so one-off events do not dominate
    q1 = (f.dropna(subset=["perf_drop_index"]).groupby("injury")
          .agg(avg_drop=("perf_drop_index", "mean"), cases=("name", "count"))
          .query("cases >= 5").sort_values("avg_drop", ascending=False))
    if not q1.empty:
        insight("Q1 · Which injuries hurt the team most?",
                f"<b>{q1.index[0]}</b> costs the team about <b>{q1.iloc[0]['avg_drop']:.2f}</b> "
                f"goals of goal difference per match ({int(q1.iloc[0]['cases'])} cases, "
                f"injury types with 5+ cases only).")
    else:
        insight("Q1 · Which injuries hurt the team most?", "Not enough data for this filter.")

    w, d, l = (f["missed_wins"].sum(), f["missed_draws"].sum(), f["missed_losses"].sum())
    total = w + d + l
    if total:
        insight("Q2 · Team record while a player was absent",
                f"<b>{int(w)}</b> wins · <b>{int(d)}</b> draws · <b>{int(l)}</b> losses "
                f"(win rate <b>{w / total:.0%}</b>).")
    else:
        insight("Q2 · Team record while a player was absent", "No match results found.")

    q3 = f.dropna(subset=["rating_change"])
    if not q3.empty:
        insight("Q3 · Do players bounce back after recovery?",
                f"Average rating change <b>{q3['rating_change'].mean():+.2f}</b>; "
                f"<b>{(q3['rating_change'] > 0).mean():.0%}</b> of returns had a better rating "
                f"than before the injury.")

    mc, cc = f["injury_month"].value_counts(), f["team_name"].value_counts()
    if not mc.empty:
        insight("Q4 · Injury clusters",
                f"Peak month is <b>{mc.index[0]}</b> ({mc.iloc[0]} injuries); "
                f"most-affected club is <b>{cc.index[0]}</b> ({cc.iloc[0]} injuries).")

    q5 = (f.dropna(subset=["perf_drop_index"]).groupby("team_name")
          .agg(drop=("perf_drop_index", "mean"), n=("name", "count")).query("n >= 10"))
    if not q5.empty:
        insight("Q5 · Which clubs suffer most?",
                f"<b>{q5['drop'].idxmax()}</b> has the largest average drop "
                f"(<b>{q5['drop'].max():.2f}</b> goals per match, clubs with 10+ records).")

    st.caption("Drop index = average goal difference in the matches before the injury minus "
               "average goal difference in the matches missed. Higher = bigger drop.")

# ----------------------------------------------------------------------------
# STEP 4: VISUALISATIONS
# ----------------------------------------------------------------------------
with tab_charts:
    # 1. Bar chart: top 10 injuries by team performance drop
    st.subheader("1. Top 10 injuries with the highest team performance drop")
    min_cases = st.slider("Minimum cases per injury type", 1, 15, 5)
    bar = (f.dropna(subset=["perf_drop_index"]).groupby("injury")
           .agg(avg_drop=("perf_drop_index", "mean"), cases=("name", "count"))
           .query("cases >= @min_cases").sort_values("avg_drop", ascending=False)
           .head(10).reset_index())
    if bar.empty:
        st.info("No injuries meet this threshold.")
    else:
        fig = px.bar(bar, x="avg_drop", y="injury", orientation="h", color="avg_drop",
                     color_continuous_scale=DROP_SCALE, hover_data=["cases"],
                     labels={"avg_drop": "Avg. goal-difference drop", "injury": "Injury"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # 2. Line chart: player timeline before / after injury
    st.subheader("2. Player performance timeline (before vs after injury)")
    cols = info["cols_found"]
    br, ar = cols["before_rating"], cols["after_rating"]
    if br and ar:
        players = sorted(f["name"].unique())
        p = st.selectbox("Player", players)
        pdf = f[f["name"] == p].sort_values("date_of_injury")
        labels = [f"{r['injury']} ({r['date_of_injury']:%b %Y})" for _, r in pdf.iterrows()]
        idx = st.selectbox("Injury", range(len(pdf)), format_func=lambda i: labels[i])
        row = pdf.iloc[idx]
        # Match1 is assumed to be the match closest to the injury, so the
        # "before" matches are reversed to read left -> right in time.
        x = [f"Before M{match_no(c)}" for c in reversed(br)] + \
            [f"After M{match_no(c)}" for c in ar]
        y = [row[c] for c in reversed(br)] + [row[c] for c in ar]
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name="Player rating",
                                   line=dict(width=3, color=GREEN), marker=dict(size=10)))
        fig.add_vline(x=len(br) - 0.5, line_dash="dash", line_color="red",
                      annotation_text="Injury")
        fig.update_layout(yaxis_title="Match rating", height=380)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    else:
        st.info("Match rating columns were not found in this dataset.")

    # 3. Heatmap: injuries by month and club
    st.subheader("3. Injury frequency across months and clubs")
    heat = pd.crosstab(f["team_name"], f["injury_month"])
    heat = heat.reindex(columns=[m for m in MONTH_ORDER if m in heat.columns])
    if heat.empty:
        st.info("No data for the heatmap.")
    else:
        fig = px.imshow(heat, aspect="auto", color_continuous_scale=HEAT_SCALE,
                        labels=dict(x="Month", y="Club", color="Injuries"), text_auto=True)
        fig.update_layout(height=max(350, 28 * len(heat)))
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # 4. Scatter: age vs drop index
    st.subheader("4. Player age vs performance drop index")
    sc = f.dropna(subset=["perf_drop_index"]).copy()
    sc["injury_days"] = sc["injury_days"].fillna(sc["injury_days"].median()).fillna(1)
    if sc.empty:
        st.info("No drop-index values for this filter.")
    else:
        fig = px.scatter(sc, x="age", y="perf_drop_index", color="position_group", color_discrete_map=POS_COLORS,
                         size="injury_days", size_max=18,
                         hover_data=["name", "team_name", "injury"],
                         labels={"perf_drop_index": "Team performance drop index",
                                 "age": "Age"})
        if len(sc) > 2 and sc["age"].nunique() > 1:  # simple least-squares trend line
            m, b = np.polyfit(sc["age"], sc["perf_drop_index"], 1)
            xs = np.array([sc["age"].min(), sc["age"].max()])
            fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                                     name="Trend", line=dict(color="#e2e8f0", dash="dash")))
            corr = sc["age"].corr(sc["perf_drop_index"])
            st.caption(f"Correlation between age and drop index: r = {corr:.2f}")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    # 5. Leaderboard: comeback players
    st.subheader("5. Comeback leaderboard (rating improvement after injury)")
    lb = (f.dropna(subset=["rating_change"]).groupby(["name", "team_name"])
          .agg(injuries=("injury", "count"),
               avg_before=("avg_rating_before", "mean"),
               avg_after=("avg_rating_after", "mean"),
               improvement=("rating_change", "mean"))
          .reset_index().sort_values("improvement", ascending=False))
    n = st.slider("Rows to show", 5, 25, 10)
    st.dataframe(
        lb.head(n).round(2).rename(columns={
            "name": "Player", "team_name": "Club", "injuries": "Injuries",
            "avg_before": "Rating before", "avg_after": "Rating after",
            "improvement": "Improvement"}),
        use_container_width=True, hide_index=True)

    # Bonus 6: results during absence
    st.subheader("Bonus: match results while a player was absent, by club")
    res = (f.groupby("team_name")[["missed_wins", "missed_draws", "missed_losses"]]
           .sum().reset_index().melt(id_vars="team_name", var_name="Result", value_name="Matches"))
    res["Result"] = res["Result"].str.replace("missed_", "").str.title()
    if res["Matches"].sum() > 0:
        fig = px.bar(res, x="team_name", y="Matches", color="Result", barmode="stack",
                     color_discrete_map={"Wins": GREEN, "Draws": AMBER, "Losses": RED},
                     labels={"team_name": "Club"})
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ----------------------------------------------------------------------------
# DATA & CLEANING SUMMARY
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("What the cleaning step did")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- Raw data: **{info['raw_rows']:,} rows x {info['raw_cols']} columns**")
        st.write(f"- Duplicate rows removed: **{info['duplicates_removed']}**")
        st.write(f"- Rows missing player/team/injury date removed: "
                 f"**{info['rows_missing_key_fields']}**")
        st.write(f"- Negative injury durations removed: **{info['negative_duration_removed']}**")
        st.write(f"- Unreadable / 'Present' return dates set to missing: "
                 f"**{info['return_date_missing']}**")
    with c2:
        st.write(f"- Missing ages filled with median: **{info['age_filled']}**")
        st.write(f"- Missing FIFA ratings filled with club median: **{info['fifa_filled']}**")
        st.write("- Dates converted to datetime; text ratings and goal differences "
                 "converted to numbers; match results standardised to Win/Draw/Loss")
        st.write(f"- Clean data: **{info['clean_rows']:,} rows**")

    st.subheader("New columns created")
    st.write("`injury_days`, `severity`, `injury_month`, `avg_rating_before`, "
             "`avg_rating_after`, `rating_change`, `avg_gd_before`, `avg_gd_missed`, "
             "`avg_gd_after`, `perf_drop_index`, `ppm_before`, `ppm_missed`, "
             "`missed_wins`, `missed_draws`, `missed_losses`")

    st.subheader("Summary statistics")
    st.dataframe(f[["age", "fifa_rating", "injury_days", "avg_rating_before",
                    "avg_rating_after", "rating_change", "perf_drop_index"]]
                 .describe().round(2), use_container_width=True)

    st.subheader("Cleaned data")
    st.dataframe(f.head(200), use_container_width=True)
    st.download_button("Download cleaned data (CSV)",
                       f.to_csv(index=False).encode("utf-8"),
                       file_name="player_injuries_clean.csv", mime="text/csv")
