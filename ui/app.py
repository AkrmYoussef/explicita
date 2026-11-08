"""Streamlit front end. It talks to the API.

Run the API first (make api), then:  streamlit run ui/app.py
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Diverging pair (blue = towards approve, red = towards reject), checked for colour-blind safety.
COLORS = {
    "light": {"approve": "#2a78d6", "reject": "#e34948"},
    "dark": {"approve": "#3987e5", "reject": "#e66767"},
}

st.set_page_config(page_title="Explicita", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        st.error(f"API error on {path}: {err}. Is the API running at {API_URL}?")
        st.stop()


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def contributions_chart(contributions: list[dict]) -> alt.Chart:
    colors = COLORS[theme()]
    df = pd.DataFrame(contributions).astype({"value": str})
    df["direction"] = df["weight"].map(
        lambda w: "Pushes towards approve" if w >= 0 else "Pushes towards reject"
    )
    order = df["feature"].tolist()  # already sorted by importance
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X("weight:Q", title="Effect on the decision"),
            y=alt.Y("feature:N", sort=order, title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Pushes towards approve", "Pushes towards reject"],
                    range=[colors["approve"], colors["reject"]],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("weight:Q", title="Weight", format="+.3f"),
            ],
        )
        .configure_axis(grid=True, gridOpacity=0.3, domain=False)
        .properties(height=max(200, 26 * len(df)))
    )


# ---- sidebar: the applicant form ---------------------------------------------
schema = api("GET", "/schema")
st.sidebar.header("Applicant")
features: dict = {}
for f in schema["features"]:
    if f["type"] == "categorical":
        options = f["options"]
        features[f["name"]] = st.sidebar.selectbox(
            f["name"], options, index=options.index(f["default"])
        )
    elif f["integer"]:
        features[f["name"]] = st.sidebar.slider(
            f["name"], int(f["min"]), int(f["max"]), int(f["default"])
        )
    else:
        features[f["name"]] = st.sidebar.slider(
            f["name"], float(f["min"]), float(f["max"]), float(f["default"])
        )

body = {"features": features}

# ---- main page ---------------------------------------------------------------
st.title("Explicita")
st.caption(
    f"Model test accuracy: {schema['metrics'].get('accuracy', '?')} · "
    f"ROC AUC: {schema['metrics'].get('roc_auc', '?')}"
)

decision = api("POST", "/predict", json=body)
label = "Approved" if decision["approved"] else "Rejected"
st.metric("Decision", label, f"{decision['probability']:.0%} chance of approval", delta_color="off")

tab_why, tab_change, tab_trust = st.tabs(["Why?", "What can I change?", "Can I trust this?"])

with tab_why:
    method = st.radio("Method", ["shap", "lime"], horizontal=True, format_func=str.upper)
    result = api("POST", f"/explain?method={method}", json=body)
    st.altair_chart(contributions_chart(result["contributions"]), width="stretch")
    with st.expander("Show as a table"):
        st.dataframe(pd.DataFrame(result["contributions"]).astype({"value": str}), hide_index=True)
    if method == "shap":
        st.caption("SHAP weights are in log-odds. Base value + sum of weights = model output.")

with tab_change:
    st.write("Small changes that would flip the decision. Age and personal status are never changed.")
    if st.button("Find changes"):
        with st.spinner("Searching..."):
            cfs = api("POST", "/counterfactuals?total=3", json=body)["counterfactuals"]
        if not cfs:
            st.info("No counterfactual found for this applicant.")
        for i, cf in enumerate(cfs, start=1):
            st.subheader(f"Option {i}: new approval chance {cf['new_probability']:.0%}")
            st.dataframe(pd.DataFrame(cf["changes"]).astype(str), hide_index=True)

with tab_trust:
    st.write("Three checks on the explanations for this applicant.")
    if st.button("Run checks"):
        with st.spinner("Running checks..."):
            report = api("POST", "/evaluate?k=3", json=body)
        rows = []
        for m in ("shap", "lime"):
            faith = report["faithfulness"][m]
            rows.append(
                {
                    "method": m.upper(),
                    "faithfulness passed": "Yes" if faith["passed"] else "No",
                    "change when top 3 removed": faith["top_k_change"],
                    "change when 3 random removed": faith["random_k_change"],
                    "stability (0-1)": report["stability"][m]["mean_top_k_overlap"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        agree = report["agreement"]
        st.write(
            f"SHAP and LIME share **{agree['top_k_overlap']:.0%}** of their top 3 features "
            f"(rank correlation {agree['rank_correlation']:.2f})."
        )
