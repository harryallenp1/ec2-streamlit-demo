"""Streamlit demo for the 'Analysing EC2 Instances' activity.

Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EC2 Instance Analysis", layout="wide")
sns.set(style="whitegrid")

COST_COLUMNS = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]


@st.cache_data
def load_and_clean(file) -> pd.DataFrame:
    data = pd.read_csv(file)
    for col in COST_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r"[$, hourly]", "", regex=True),
                errors="coerce",
            )
    data["Instance Memory"] = pd.to_numeric(
        data["Instance Memory"].astype(str).str.replace(" GiB", "", regex=False),
        errors="coerce",
    )
    data["vCPUs"] = pd.to_numeric(
        data["vCPUs"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    return data


def detect_outliers(data: pd.DataFrame, column: str) -> pd.DataFrame:
    q1, q3 = data[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data[column] < lower) | (data[column] > upper)]


st.title("Amazon EC2 Instance Cost Analysis")

BUNDLED_CSV = Path(__file__).parent / "ec2dataset.csv"

uploaded = st.sidebar.file_uploader(
    "Optional: upload a different EC2 CSV", type="csv"
)
if uploaded is not None:
    source = uploaded
    st.sidebar.success("Using uploaded file")
elif BUNDLED_CSV.exists():
    source = str(BUNDLED_CSV)
    st.sidebar.caption("Using bundled dataset: ec2dataset.csv")
else:
    st.info("Upload the EC2 CSV in the sidebar to begin.")
    st.stop()

data = load_and_clean(source)
cost_cols = [c for c in COST_COLUMNS if c in data.columns]

tab1, tab2, tab3 = st.tabs(
    ["Part 1 · Explore costs", "Part 1 · Instance families", "Part 2 · Regression"]
)

# ---------------- Part 1: explore ----------------
with tab1:
    st.subheader("Dataset preview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", len(data))
    c2.metric("Columns", data.shape[1])
    st.dataframe(data.head(20), width="stretch")

    st.subheader("Missing values after cleaning")
    st.dataframe(data[cost_cols].isnull().sum().rename("Missing").to_frame())

    st.subheader("Summary statistics (hourly USD)")
    st.dataframe(data[cost_cols].describe(), width="stretch")

    st.subheader("Cost distribution by pricing option")
    log_scale = st.checkbox("Log scale (big instances squash the plot otherwise)", True)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=data[cost_cols], palette="Set2", ax=ax)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Cost (USD / hour)")
    ax.set_title("Cost Comparison of Amazon EC2 Instances (Hourly)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig)

    st.subheader("Outliers (IQR method)")
    outlier_col = st.selectbox("Cost column", cost_cols)
    outliers = detect_outliers(data, outlier_col)
    st.write(f"**{len(outliers)}** outliers in *{outlier_col}*")
    st.dataframe(
        outliers[["Name", "Instance Memory", "vCPUs", outlier_col]]
        .sort_values(outlier_col, ascending=False),
        width="stretch",
    )

    st.subheader("Reserved vs On-Demand savings")
    comp = data[["Name", "On Demand", "Linux Reserved cost"]].dropna()
    comp = comp[comp["On Demand"] > 0].copy()
    comp["Savings %"] = (
        (comp["On Demand"] - comp["Linux Reserved cost"]) / comp["On Demand"] * 100
    )
    st.metric("Average savings from Linux Reserved", f"{comp['Savings %'].mean():.1f}%")
    st.write("10 cheapest instances (On Demand):")
    st.dataframe(comp.sort_values("On Demand").head(10), width="stretch")

    st.subheader("Most cost-effective (cost per vCPU / per GiB)")
    eff = data[["Name", "vCPUs", "Instance Memory", "On Demand"]].dropna().copy()
    eff = eff[(eff["vCPUs"] > 0) & (eff["Instance Memory"] > 0)]
    eff["$ per vCPU"] = eff["On Demand"] / eff["vCPUs"]
    eff["$ per GiB"] = eff["On Demand"] / eff["Instance Memory"]
    metric = st.radio("Rank by", ["$ per vCPU", "$ per GiB"], horizontal=True)
    st.dataframe(eff.sort_values(metric).head(10), width="stretch")

# ---------------- Part 1: families ----------------
with tab2:
    data["Family"] = data["Name"].astype(str).str.split().str[0]
    families = sorted(data["Family"].dropna().unique())
    default = [f for f in ("T2", "T3") if f in families] or families[:2]
    chosen = st.multiselect("Instance families", families, default=default)

    for fam in chosen:
        fam_df = data[data["Family"] == fam]
        st.markdown(f"### {fam} — {len(fam_df)} instances")
        st.dataframe(fam_df[cost_cols].describe(), width="stretch")
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.boxplot(data=fam_df[cost_cols], palette="Blues", showmeans=True, ax=ax)
        ax.set_title(f"Cost Distribution for {fam} Instances")
        ax.set_ylabel("Cost (USD / hour)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)

    if chosen:
        st.markdown("### On-Demand vs Reserved across selected families")
        fam_comp = (
            data[data["Family"].isin(chosen)][
                ["Name", "Family", "On Demand", "Linux Reserved cost"]
            ]
            .dropna()
            .sort_values("On Demand")
        )
        st.dataframe(fam_comp.head(10), width="stretch")

# ---------------- Part 2: regression ----------------
with tab3:
    clean = data.dropna(subset=["On Demand", "Instance Memory", "vCPUs"])
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    X = clean[["Instance Memory", "vCPUs"]]
    y = clean["On Demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"Training samples: **{len(X_train)}** · Testing samples: **{len(X_test)}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Intercept", f"{model.intercept_:.4f}")
    c2.metric("$ per GiB memory", f"{model.coef_[0]:.4f}")
    c3.metric("$ per vCPU", f"{model.coef_[1]:.4f}")

    mse = mean_squared_error(y_test, y_pred)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
    m2.metric("MSE", f"{mse:.4f}")
    m3.metric("RMSE", f"{mse ** 0.5:.4f}")
    m4.metric("R²", f"{r2_score(y_test, y_pred):.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.7, color="b")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--")
    ax.set_title("Actual vs Predicted On-Demand Costs")
    ax.set_xlabel("Actual On-Demand Cost")
    ax.set_ylabel("Predicted On-Demand Cost")
    st.pyplot(fig)

    st.subheader("Predict a new instance")
    p1, p2 = st.columns(2)
    mem = p1.number_input("Memory (GiB)", min_value=0.5, value=4.0, step=0.5)
    cpu = p2.number_input("vCPUs", min_value=1, value=2, step=1)
    new = pd.DataFrame([[mem, cpu]], columns=["Instance Memory", "vCPUs"])
    pred = model.predict(new)[0]
    if pred < 0:
        st.warning(
            f"Predicted On-Demand cost: **${pred:.4f} / hour** — negative! "
            "The model's intercept is pulled below zero by a few huge instances "
            "(e.g. U7IN-32TB at $407/hr), so a straight line fits small instances badly. "
            "A log-scale model or removing outliers would fix this."
        )
    else:
        st.success(f"Predicted On-Demand cost: **${pred:.4f} / hour**")
