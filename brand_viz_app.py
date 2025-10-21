# brand_viz_app.py
import pandas as pd
import streamlit as st
import altair as alt

# ---------- CONFIG ----------
st.set_page_config(page_title="SERP Brand Landscape", layout="wide")
st.title("🎰 SERP Brand Landscape Explorer")
pos_filter = st.slider("Filter positions", 1, 10, (1, 4))

@st.cache_data
def load_data():
    df = pd.read_csv("output/brand_classifications_enriched.csv")
    df["classification"] = df["classification"].fillna("Other")
    df["state"] = df["state"].fillna("Unknown")
    # Weight by visibility: position 1=10, 10=1, anything >10 gets weight 0
    df["position_weight"] = df["position"].apply(lambda p: max(0, 11 - p) if pd.notna(p) else 0)
    return df

df = load_data()
df = df[df["position"].between(pos_filter[0], pos_filter[1])]

# ---------- SIDEBAR ----------
st.sidebar.header("🔍 Filters")
states = sorted(df["state"].unique())
state_sel = st.sidebar.multiselect("Select States", states, default=states)
keyword_sel = st.sidebar.text_input("Filter by keyword (optional)").strip().lower()
metric_sel = st.sidebar.radio("Metric", ["Raw Count", "Position-Weighted Share"])
view = st.sidebar.radio("View Mode", ["Overview", "Brand Breakdown", "Keyword Detail", "SERP Table"])

df_filt = df[df["state"].isin(state_sel)]
if keyword_sel:
    df_filt = df_filt[df_filt["keyword"].str.lower().str.contains(keyword_sel, na=False)]

# ---------- OVERVIEW ----------
if view == "Overview":
    st.header("🏛️ Classification Share by State")

    # --- Weighted summary ---
    if metric_sel == "Position-Weighted Share":
        summary = (
            df_filt.groupby(["state", "classification"])["position_weight"]
            .sum()
            .reset_index(name="weight")
        )
        total = summary.groupby("state")["weight"].sum().reset_index(name="total")
        summary = summary.merge(total, on="state")
        summary["share"] = (summary["weight"] / summary["total"] * 100).round(1)
        y_field = "weight:Q"
        title_suffix = " (Position-Weighted)"
    else:
        summary = df_filt.groupby(["state", "classification"]).size().reset_index(name="count")
        total = summary.groupby("state")["count"].sum().reset_index(name="total")
        summary = summary.merge(total, on="state")
        summary["share"] = (summary["count"] / summary["total"] * 100).round(1)
        y_field = "count:Q"
        title_suffix = " (Raw Counts)"

    # --- Chart ---
    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("state:N", title="State", sort=states),
            y=alt.Y(y_field, title=metric_sel),
            color=alt.Color(
                "classification:N",
                scale=alt.Scale(
                    domain=["Real", "Sweeps", "Both", "Other"],
                    range=["#2ecc71", "#3498db", "#f1c40f", "#bdc3c7"]
                ),
                title="Classification"
            ),
            tooltip=["state", "classification", "share"]
        )
        .properties(height=600)
    )

    st.subheader(f"Stacked Share by Classification {title_suffix}")
    st.altair_chart(chart.encode(y=alt.Y("share:Q", stack="normalize", title="Share (%)")), use_container_width=True)
    st.dataframe(
        summary[["state", "classification", "share"]]
        .sort_values(["state", "share"], ascending=[True, False]),
        use_container_width=True
    )

# ---------- BRAND BREAKDOWN ----------
elif view == "Brand Breakdown":
    st.header("🏆 Top Brand Mentions")

    col1, col2 = st.columns(2)

    # ---- Real Money Brands ----
    with col1:
        st.subheader("Real Money Brands (Top 25)")
        real = (
            df_filt.assign(brand=df_filt["real_brands"].str.split(", "))
            .explode("brand")
            .query("brand != ''")
            .groupby("brand")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(25)
        )

        chart_real = (
            alt.Chart(real)
            .mark_bar()
            .encode(
                y=alt.Y("brand:N", sort="-x", title="Brand"),
                x=alt.X("count:Q", title="Mentions"),
                color=alt.value("#2ecc71"),
                tooltip=["brand", "count"]
            )
            .properties(height=500)
        )
        st.altair_chart(chart_real, use_container_width=True)

    # ---- Sweepstakes Brands ----
    with col2:
        st.subheader("Sweepstakes Brands (Top 25)")
        sweeps = (
            df_filt.assign(brand=df_filt["sweeps_brands"].str.split(", "))
            .explode("brand")
            .query("brand != ''")
            .groupby("brand")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(25)
        )

        chart_sweeps = (
            alt.Chart(sweeps)
            .mark_bar()
            .encode(
                y=alt.Y("brand:N", sort="-x", title="Brand"),
                x=alt.X("count:Q", title="Mentions"),
                color=alt.value("#3498db"),
                tooltip=["brand", "count"]
            )
            .properties(height=500)
        )
        st.altair_chart(chart_sweeps, use_container_width=True)

    # Combined leaderboard (optional)
    st.subheader("🥇 Combined Leaderboard")
    combined = (
        pd.concat([
            real.assign(type="Real"),
            sweeps.assign(type="Sweeps")
        ])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(combined, use_container_width=True)


# ---------- KEYWORD DETAIL ----------
elif view == "Keyword Detail":
    st.header("🔎 Keyword-Level Analysis")

    # ensure numeric & drop nulls
    df_kw = df_filt.dropna(subset=["keyword", "position_weight"]).copy()

    # group by keyword + classification
    kw_summary = (
        df_kw.groupby(["keyword", "classification"])["position_weight"]
        .sum()
        .reset_index(name="weight")
    )

    # normalise shares within each keyword group (transform keeps index)
    kw_summary["share"] = (
        kw_summary["weight"]
        / kw_summary.groupby("keyword")["weight"].transform("sum")
        * 100
    ).round(1)

    # limit axis to 0–100 for clean rendering
    x_scale = alt.Scale(domain=[0, 100])

    st.subheader("Weighted Distribution by Keyword (0–100 %)")
    chart_kw = (
        alt.Chart(kw_summary)
        .mark_bar()
        .encode(
            y=alt.Y("keyword:N", sort="-x", title="Keyword"),
            x=alt.X("share:Q", title="Weighted Share (%)", scale=x_scale),
            color=alt.Color(
                "classification:N",
                scale=alt.Scale(
                    domain=["Real", "Sweeps", "Both", "Other"],
                    range=["#2ecc71", "#3498db", "#f1c40f", "#bdc3c7"],
                ),
            ),
            tooltip=["keyword", "classification", "share"],
        )
        .properties(height=500)
    )

    st.altair_chart(chart_kw, use_container_width=True)

    # sanity check table (each keyword sums to ~100)
    check = (
        kw_summary.groupby("keyword")["share"].sum().reset_index(name="sum_share")
        .sort_values("sum_share", ascending=False)
    )
    if any(check["sum_share"] > 100.1):
        st.warning("⚠️ Some keywords exceed 100 % total share due to rounding.")

    st.dataframe(
        kw_summary.sort_values(["keyword", "share"], ascending=[True, False]),
        use_container_width=True,
    )


# ---------- SERP TABLE ----------
else:
    st.header("🧾 Full SERP Classification Table")

    st.dataframe(
        df_filt[["state", "keyword", "position", "position_weight", "url", "classification", "real_brands", "sweeps_brands"]]
        .sort_values(["state", "keyword", "position"]),
        use_container_width=True,
        height=800
    )
