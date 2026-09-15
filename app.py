"""
app.py
------
Streamlit UI. Deliberately thin: every number and chart here is computed by
src/data.py, src/topic_model.py, src/stats_engine.py, and src/viz.py -
this file only lays out widgets and calls those modules.

EXECUTION NOTE: Streamlit and Plotly are not installed in the sandbox that
built this project (pip to pypi.org was blocked there - see README.md for
the full explanation). This file was written against Streamlit's and
Plotly's documented APIs but not launched/rendered in that sandbox. Once
you `pip install -r requirements.txt` in a normal environment, run:

    streamlit run app.py

Run `python scripts/fetch_dataset.py` then `python scripts/run_pipeline.py`
first if data/processed/ doesn't exist yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import stats_engine, viz
from src.data import CATEGORY_COL, DATE_COL, load_processed

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"

st.set_page_config(page_title="NLP Customer Sentiment Dashboard", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_processed()


@st.cache_data
def get_topics() -> list[dict]:
    with open(PROCESSED_DIR / "topics.json") as f:
        return json.load(f)


@st.cache_data
def get_stats_results() -> dict:
    with open(PROCESSED_DIR / "stats_results.json") as f:
        return json.load(f)


def main() -> None:
    st.title("NLP-Driven Customer Sentiment Dashboard")
    st.caption(
        "Twitter US Airline Sentiment dataset - transformer-based classification, "
        "unsupervised topic modeling, and a real statistical significance layer. "
        "See README.md for what ran in the build sandbox vs. what runs here."
    )

    try:
        df = get_data()
        topics = get_topics()
        stats_results = get_stats_results()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    segments = sorted(df[CATEGORY_COL].unique().tolist())
    selected_segments = st.sidebar.multiselect("Segments", segments, default=segments)
    date_min, date_max = df[DATE_COL].min(), df[DATE_COL].max()
    st.sidebar.caption(f"Data range: {date_min.date()} to {date_max.date()}")

    filtered = df[df[CATEGORY_COL].isin(selected_segments)]

    label_col = "gold_sentiment" if "model_sentiment" not in filtered.columns else "model_sentiment"
    if label_col == "gold_sentiment":
        st.sidebar.info(
            "Showing the dataset's real human-annotated sentiment label. "
            "Run scripts/classify_with_model.py (needs torch+transformers) "
            "to populate model_sentiment and switch this view to live "
            "transformer predictions."
        )

    tab_overview, tab_segments, tab_trends, tab_topics, tab_stats = st.tabs(
        ["Overview", "Sentiment by Segment", "Trends", "Topics", "Statistical Significance"]
    )

    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total reviews", f"{len(filtered):,}")
        col2.metric("Segments", len(selected_segments))
        col3.metric(
            "Positive share",
            f"{(filtered[label_col] == 'positive').mean():.1%}",
        )
        col4.metric(
            "Negative share",
            f"{(filtered[label_col] == 'negative').mean():.1%}",
        )
        st.plotly_chart(
            viz.sentiment_by_segment_bar(filtered, label_col, CATEGORY_COL),
            use_container_width=True,
            key="overview_sentiment_by_segment",
        )

    with tab_segments:
        st.plotly_chart(
            viz.sentiment_by_segment_bar(filtered, label_col, CATEGORY_COL),
            use_container_width=True,
            key="segments_sentiment_by_segment",
        )
        st.dataframe(
            filtered.groupby(CATEGORY_COL)["sentiment_score"].agg(["mean", "std", "count"]).sort_values("mean"),
            use_container_width=True,
        )

    with tab_trends:
        freq = st.radio("Aggregate by", ["D", "W"], horizontal=True, format_func=lambda f: "Day" if f == "D" else "Week")
        st.plotly_chart(
            viz.sentiment_trend_line(filtered, DATE_COL, "sentiment_score", CATEGORY_COL, freq=freq),
            use_container_width=True,
            key=f"trend_line_{freq}",
        )

    with tab_topics:
        st.plotly_chart(viz.topic_prevalence_bar(topics), use_container_width=True, key="topic_prevalence")
        st.plotly_chart(
            viz.topic_by_segment_heatmap(filtered, "topic_label", CATEGORY_COL),
            use_container_width=True,
            key="topic_by_segment_heatmap",
        )
        st.subheader("Top terms per topic")
        for t in sorted(topics, key=lambda x: -x["prevalence"]):
            st.markdown(f"**{t['label']}** ({t['prevalence']:.1%}) - {', '.join(t['top_terms'])}")

    with tab_stats:
        st.plotly_chart(viz.stats_effect_size_bar(stats_results), use_container_width=True, key="stats_effect_size")
        for name, result in stats_results.items():
            with st.expander(f"{result['test_name']} - {name}"):
                st.json(result)

        st.subheader("Run a custom comparison")
        col_a, col_b = st.columns(2)
        seg_a = col_a.selectbox("Segment A", segments, index=0)
        seg_b = col_b.selectbox("Segment B", segments, index=min(1, len(segments) - 1))
        if seg_a != seg_b:
            values_a = filtered.loc[filtered[CATEGORY_COL] == seg_a, "sentiment_score"]
            values_b = filtered.loc[filtered[CATEGORY_COL] == seg_b, "sentiment_score"]
            custom_result = stats_engine.compare_two_groups(values_a, values_b, seg_a, seg_b)
            st.write(f"**{custom_result.test_name}**: t={custom_result.statistic:.4f}, "
                     f"p={custom_result.p_value:.4g}, Cohen's d={custom_result.effect_size:.4f}")
            st.write(custom_result.verdict)

        st.divider()
        st.subheader("Executive summary (optional, Claude API)")
        try:
            from src.summary import generate_summary

            top_topic = sorted(topics, key=lambda x: -x["prevalence"])[0]
            anova = stats_results.get("anova_all_airlines", {})
            if anova:
                summary_text = generate_summary(
                    topic_label=top_topic["label"],
                    prevalence=top_topic["prevalence"],
                    group_a=anova["groups"][0],
                    group_b=anova["groups"][-1],
                    test_name=anova["test_name"],
                    p_value=anova["p_value"],
                    effect_size=anova["effect_size"],
                    effect_size_name=anova["effect_size_name"],
                    verdict=anova["verdict"],
                )
                st.info(summary_text)
        except Exception as exc:  # pragma: no cover - optional feature, fail soft
            st.caption(f"Executive summary unavailable: {exc}")


if __name__ == "__main__":
    main()
