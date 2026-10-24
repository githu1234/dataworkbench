"""DataWorkbench: one local app for data engineering, analysis, and machine learning."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataWorkbench", page_icon="◈", layout="wide")

DEMO_DATA = pd.DataFrame({
    "order_date": pd.date_range("2025-01-01", periods=32, freq="D").astype(str),
    "region": ["North", "West", "South", "East"] * 8,
    "channel": ["Organic", "Paid", "Referral", "Paid", "Organic", "Referral", "Paid", "Organic"] * 4,
    "sessions": [820,640,510,730,910,455,680,990,760,590,840,520,1050,710,490,880,660,940,570,770,1120,620,860,540,930,695,475,1010,735,825,580,960],
    "conversions": [42,29,24,38,53,21,36,59,40,27,45,25,61,37,23,51,34,55,26,41,66,30,47,26,54,39,22,58,36,44,28,57],
    "revenue": [2460,1740,1330,2240,3190,1210,2120,3480,2380,1580,2740,1490,3670,2190,1320,3020,1990,3320,1510,2470,3890,1810,2860,1530,3260,2310,1250,3510,2140,2660,1640,3410],
    "ad_spend": [420,580,160,510,460,140,600,390,430,540,470,155,630,590,145,440,610,405,170,520,650,570,480,150,455,620,135,425,585,465,165,445],
    "customer_rating": [4.2,3.8,4.0,4.4,4.6,3.7,4.1,4.8,4.3,3.9,4.5,3.8,4.7,4.2,3.9,4.6,4.0,4.7,3.8,4.4,4.9,4.1,4.6,3.9,4.5,4.3,3.7,4.8,4.2,4.5,3.9,4.7],
})
DEMO_DATA.loc[5, "customer_rating"] = np.nan
DEMO_DATA.loc[17, "ad_spend"] = np.nan
DEMO_DATA = pd.concat([DEMO_DATA, DEMO_DATA.iloc[[3]]], ignore_index=True)


def clean_names(columns: list[Any]) -> list[str]:
    names = [re.sub(r"[^a-zA-Z0-9]+", "_", str(c).strip().lower()).strip("_") or "column" for c in columns]
    counts: dict[str, int] = {}
    unique = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        unique.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return unique


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def transform_data(raw: pd.DataFrame, standardize: bool, dedupe: bool, missing: str,
                   keep: list[str], filter_col: str | None, low: float | None, high: float | None) -> pd.DataFrame:
    frame = raw.copy()
    mapping = dict(zip(raw.columns, clean_names(list(raw.columns)))) if standardize else {c: c for c in raw.columns}
    frame = frame.rename(columns=mapping)
    if keep:
        frame = frame[[mapping[c] for c in keep if c in mapping]]
    if filter_col:
        name = mapping[filter_col]
        values = pd.to_numeric(frame[name], errors="coerce")
        if low is not None:
            frame = frame[values >= low]
        if high is not None:
            frame = frame[pd.to_numeric(frame[name], errors="coerce") <= high]
    if missing == "Drop rows with missing values":
        frame = frame.dropna()
    elif missing == "Fill numeric nulls with mean":
        nums = frame.select_dtypes(include="number").columns
        frame.loc[:, nums] = frame.loc[:, nums].fillna(frame.loc[:, nums].mean())
    if dedupe:
        frame = frame.drop_duplicates()
    return frame.reset_index(drop=True)


def sql_for(source: str, raw_cols: list[str], numeric_cols: list[str], standardize: bool, dedupe: bool, missing: str,
            keep: list[str], filter_col: str | None, low: float | None, high: float | None) -> str:
    names = clean_names(raw_cols) if standardize else raw_cols
    numeric_names = [names[raw_cols.index(c)] for c in numeric_cols if c in raw_cols]
    chosen_raw = keep or raw_cols
    chosen = [names[raw_cols.index(c)] for c in chosen_raw if c in raw_cols]
    expressions = []
    for name in chosen:
        col = quote(name)
        if missing == "Fill numeric nulls with mean" and name in numeric_names:
            expressions.append(f"COALESCE({col}, AVG({col}) OVER ()) AS {col}")
        else:
            expressions.append(col)
    table = re.sub(r"[^a-zA-Z0-9_]+", "_", source.rsplit(".", 1)[0]) or "uploaded_data"
    query = "SELECT " + ("DISTINCT " if dedupe else "") + ",\n       ".join(expressions) + f"\nFROM {table}"
    conditions = []
    if missing == "Drop rows with missing values":
        conditions.extend(f"{quote(c)} IS NOT NULL" for c in chosen)
    if filter_col:
        name = names[raw_cols.index(filter_col)]
        if low is not None:
            conditions.append(f"{quote(name)} >= {low:g}")
        if high is not None:
            conditions.append(f"{quote(name)} <= {high:g}")
    if conditions:
        query += "\nWHERE " + "\n  AND ".join(conditions)
    return query + ";"


def profile(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No rows to display. Adjust the pipeline or upload another dataset.")
        return
    completeness = 100 * (1 - frame.isna().sum().sum() / max(frame.size, 1))
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(frame):,}")
    b.metric("Columns", f"{frame.shape[1]:,}")
    c.metric("Missing cells", f"{frame.isna().sum().sum():,}")
    d.metric("Completeness", f"{completeness:.1f}%")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Column profile")
        rows = []
        for name in frame.columns:
            series = frame[name]
            numeric = pd.api.types.is_numeric_dtype(series)
            rows.append({
                "Column": name,
                "Type": "numeric" if numeric else "text / category",
                "Filled": f"{series.notna().mean() * 100:.1f}%",
                "Unique": int(series.nunique(dropna=True)),
                "Missing": int(series.isna().sum()),
                "Mean": round(float(series.mean()), 3) if numeric and series.notna().any() else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Numeric overview")
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(numeric.describe().T.round(2), use_container_width=True)
    st.markdown("#### Data preview")
    st.dataframe(frame.head(12), use_container_width=True, hide_index=True)


def main() -> None:
    st.markdown("""
    <style>
    .stApp { background: #f5f7fb; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    h1 { color: #17243a; }
    [data-testid="stSidebar"] { background: #101a2c; }
    [data-testid="stSidebar"] * { color: #e5ebf5; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e7ebf3; padding: 16px 18px; border-radius: 14px; }
    [data-testid="stMetricLabel"] { color: #6b7890; }
    [data-testid="stMetricValue"] { color: #17243a; }
    .eyebrow { color: #687792; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .11em; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ◈ DataWorkbench")
        st.caption("One workspace for your data")
        st.divider()
        upload = st.file_uploader("Upload a dataset", type=["csv", "tsv", "txt"])
        if upload is not None:
            try:
                raw = pd.read_csv(upload, sep="\t" if upload.name.lower().endswith(".tsv") else ",")
                source = upload.name
                st.success(f"Loaded {len(raw):,} rows · {len(raw.columns)} columns")
            except Exception as error:
                st.error(f"Could not read this file: {error}")
                raw, source = DEMO_DATA.copy(), "demo_marketing_data.csv"
        else:
            raw, source = DEMO_DATA.copy(), "demo_marketing_data.csv"
            st.info("Showing sample data. Upload a CSV to start with yours.")
        st.divider()
        page = st.radio("Workspace", ["Overview", "Data Engineering", "Data Analysis", "Data Science"], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Data is processed in this app session.")

    st.markdown('<div class="eyebrow">YOUR DATA WORKSPACE</div>', unsafe_allow_html=True)
    st.title(page)
    st.caption("Turn a dataset into a clean pipeline, useful insights, and a predictive model.")

    numeric_cols = list(raw.select_dtypes(include="number").columns)

    if page == "Overview":
        st.markdown("#### Start with the shape and quality of your data")
        profile(raw)
        st.markdown("#### Your workspace")
        x, y, z = st.columns(3)
        x.markdown("**🛠 Data Engineering**\n\nClean columns, handle missing values, filter records, remove duplicates, and export SQL.")
        y.markdown("**📊 Data Analysis**\n\nCompare categories, explore relationships, and review correlations and statistics.")
        z.markdown("**🧠 Data Science**\n\nTrain regression or classification models and inspect held-out performance.")

    elif page == "Data Engineering":
        st.caption("Configure a repeatable cleaning pipeline. Preview the output and SQL before downloading.")
        with st.container(border=True):
            st.markdown("#### Pipeline steps")
            left, right = st.columns(2)
            with left:
                standardize = st.checkbox("Standardize column names", value=True)
                dedupe = st.checkbox("Remove duplicate rows", value=True)
                missing = st.selectbox("Missing values", ["Leave as-is", "Drop rows with missing values", "Fill numeric nulls with mean"], index=2)
            with right:
                keep = st.multiselect("Keep columns", list(raw.columns), default=list(raw.columns))
                filter_col = st.selectbox("Optional numeric filter", ["None"] + numeric_cols)
                filter_col = None if filter_col == "None" else filter_col
                low = high = None
                if filter_col:
                    vals = pd.to_numeric(raw[filter_col], errors="coerce").dropna()
                    if not vals.empty:
                        c1, c2 = st.columns(2)
                        low = c1.number_input("Minimum", value=float(vals.min()), key="filter_low")
                        high = c2.number_input("Maximum", value=float(vals.max()), key="filter_high")
            output = transform_data(raw, standardize, dedupe, missing, keep, filter_col, low, high)
        st.markdown("#### Pipeline result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Input rows", f"{len(raw):,}")
        c2.metric("Output rows", f"{len(output):,}")
        c3.metric("Rows removed", f"{len(raw) - len(output):,}")
        st.dataframe(output.head(15), use_container_width=True, hide_index=True)
        st.download_button("Download cleaned CSV", output.to_csv(index=False).encode("utf-8"), "cleaned_data.csv", "text/csv", type="primary")
        sql = sql_for(source, list(raw.columns), numeric_cols, standardize, dedupe, missing, keep, filter_col, low, high)
        with st.expander("View generated SQL"):
            st.code(sql, language="sql")
            st.download_button("Download SQL", sql.encode("utf-8"), "data_pipeline.sql", "text/plain")

    elif page == "Data Analysis":
        st.caption("Build a visual from your data, then inspect numeric relationships.")
        if raw.empty or not len(raw.columns):
            st.warning("Upload a dataset with columns to begin.")
        else:
            x, y, z = st.columns(3)
            x_col = x.selectbox("Group or X axis", list(raw.columns))
            y_choices = numeric_cols or ["Count rows"]
            y_col = y.selectbox("Measure or Y axis", y_choices)
            agg = z.selectbox("Aggregation", ["Sum", "Average", "Count"] if numeric_cols else ["Count"])
            frame = raw.copy()
            if y_col == "Count rows":
                result = frame.groupby(x_col, dropna=False).size().sort_values(ascending=False).head(25)
                st.bar_chart(result, use_container_width=True)
            else:
                if x_col == y_col and len(raw.columns) > 1:
                    x_col = next(c for c in raw.columns if c != y_col)
                frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
                if pd.api.types.is_numeric_dtype(frame[x_col]) and x_col != y_col:
                    chart = st.radio("Chart", ["Scatter", "Line"], horizontal=True)
                    points = frame[[x_col, y_col]].dropna().sort_values(x_col)
                    if chart == "Scatter":
                        st.scatter_chart(points, x=x_col, y=y_col, use_container_width=True)
                    else:
                        st.line_chart(points.set_index(x_col)[y_col], use_container_width=True)
                else:
                    grouped = frame.groupby(x_col, dropna=False)[y_col]
                    result = grouped.sum(min_count=1) if agg == "Sum" else grouped.mean() if agg == "Average" else grouped.count()
                    st.bar_chart(result.sort_values(ascending=False).head(25), use_container_width=True)
            st.markdown("#### Correlations")
            corr = raw.select_dtypes(include="number").corr(numeric_only=True)
            st.dataframe(corr.round(3), use_container_width=True) if not corr.empty else st.info("Add numeric columns to see correlations.")
            st.markdown("#### Summary statistics")
            if numeric_cols:
                st.dataframe(raw[numeric_cols].describe().T.round(2), use_container_width=True)

    elif page == "Data Science":
        st.caption("Train a model and evaluate it on data the model did not see during training.")
        if len(raw) < 5 or len(raw.columns) < 2:
            st.warning("Upload at least five rows and two columns to train a model.")
        else:
            target = st.selectbox("Target to predict", list(raw.columns))
            candidates = [c for c in raw.columns if c != target]
            features = st.multiselect("Input features", candidates, default=candidates)
            model_choice = st.selectbox("Model", ["Random forest", "Linear / logistic regression"])
            test_pct = st.slider("Test data", 10, 40, 20, 5)
            if not features:
                st.info("Choose at least one input feature.")
            elif st.button("Train model", type="primary"):
                try:
                    from sklearn.compose import ColumnTransformer
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    from sklearn.impute import SimpleImputer
                    from sklearn.linear_model import LogisticRegression, Ridge
                    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
                    from sklearn.model_selection import train_test_split
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import OneHotEncoder, StandardScaler

                    data = raw[features + [target]].dropna(subset=[target]).copy()
                    y = data[target]
                    classification = (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= min(10, max(2, int(len(y) * .08)))
                    if classification and y.nunique() < 2:
                        st.error("The target needs at least two distinct classes.")
                        return
                    stratify = y if classification and y.value_counts().min() >= 2 else None
                    x_train, x_test, y_train, y_test = train_test_split(
                        data[features], y, test_size=test_pct / 100, random_state=42, stratify=stratify
                    )
                    num_features = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
                    cat_features = [c for c in features if c not in num_features]
                    transformers = []
                    if num_features:
                        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_features))
                    if cat_features:
                        transformers.append(("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat_features))
                    prep = ColumnTransformer(transformers)
                    if classification:
                        estimator = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced") if model_choice == "Random forest" else LogisticRegression(max_iter=1500)
                    else:
                        estimator = RandomForestRegressor(n_estimators=180, random_state=42) if model_choice == "Random forest" else Ridge()
                    model = Pipeline([("prepare", prep), ("model", estimator)])
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    st.markdown("#### Held-out performance")
                    a, b, c = st.columns(3)
                    if classification:
                        a.metric("Task", "Classification")
                        b.metric("Accuracy", f"{accuracy_score(y_test, prediction):.3f}")
                        c.metric("Weighted F1", f"{f1_score(y_test, prediction, average='weighted', zero_division=0):.3f}")
                    else:
                        a.metric("Task", "Regression")
                        b.metric("R²", f"{r2_score(y_test, prediction):.3f}" if len(y_test) > 1 else "—")
                        c.metric("Mean absolute error", f"{mean_absolute_error(y_test, prediction):.3f}")
                        st.caption(f"RMSE: {mean_squared_error(y_test, prediction) ** .5:.3f}")
                    st.caption(f"Trained with {len(x_train):,} rows; evaluated on {len(x_test):,} rows. Model: {model_choice}.")
                    fitted = model.named_steps["model"]
                    if hasattr(fitted, "feature_importances_"):
                        names = model.named_steps["prepare"].get_feature_names_out()
                        importance = pd.Series(fitted.feature_importances_, index=names).sort_values(ascending=False).head(20)
                        st.markdown("#### Most influential features")
                        st.bar_chart(importance, horizontal=True, use_container_width=True)
                    st.markdown("#### Test predictions")
                    st.dataframe(pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": prediction}).head(100), use_container_width=True, hide_index=True)
                except Exception as error:
                    st.error(f"Model training couldn't finish: {error}")
                    st.caption("Try more rows, another target, or fewer features.")


if __name__ == "__main__":
    main()



