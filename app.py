import glob
import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KAGGLE_SLUG = "sonalshinde123/netflix-customer-churn-dataset"
LOCAL_CSV = "netflix_customer_churn.csv"

CHOC = "#6B3A1E"
GOLD = "#D4AF37"
CREAM = "#FBF6EC"
INK = "#3E2412"
BANNER = "#0D0D0D"

st.set_page_config(page_title="Netflix Churn Dashboard", page_icon="🎬", layout="wide")

st.markdown(
    f"""
<style>
.stApp {{ background-color: #FFFFFF; }}
section[data-testid="stSidebar"] {{ background-color: {CREAM}; border-right: 3px solid {GOLD}; }}
h2, h3, h4, p, label, span, .stMarkdown, .stCaption {{ color: {INK}; }}
.banner {{ background: {BANNER}; border-bottom: 5px solid {GOLD}; border-radius: 14px;
          padding: 24px 28px; margin-bottom: 18px; }}
.banner-title {{ color: #FFFFFF; font-size: 34px; font-weight: 800; line-height: 1.2; }}
.banner-sub {{ color: {GOLD}; font-size: 15px; margin-top: 6px; }}
.kpi-choc {{ background: {CHOC}; border-radius: 12px; padding: 18px; text-align: center; }}
.kpi-gold {{ background: {GOLD}; border-radius: 12px; padding: 18px; text-align: center; }}
.kpi-cream {{ background: {CREAM}; border: 2px solid {CHOC}; border-radius: 12px; padding: 18px; text-align: center; }}
.kpi-title {{ font-size: 13px; letter-spacing: 1px; text-transform: uppercase; }}
.kpi-value {{ font-size: 30px; font-weight: 800; }}
.kpi-choc .kpi-title, .kpi-choc .kpi-value {{ color: #FFFFFF; }}
.kpi-gold .kpi-title, .kpi-gold .kpi-value {{ color: {BANNER}; }}
.kpi-cream .kpi-title, .kpi-cream .kpi-value {{ color: {CHOC}; }}
.stTabs [aria-selected="true"] {{ color: {CHOC} !important; border-bottom: 3px solid {GOLD}; }}
.stButton > button {{ background-color: {CHOC}; color: {GOLD}; border: 2px solid {GOLD}; font-weight: 700; }}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(title, value, style):
    st.markdown(
        f'<div class="{style}"><div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def new_fig(width=6, height=3.8):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor(CREAM)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(CHOC)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(GOLD)
    return fig, ax


# ---------------------------------------------------------------- banner
st.markdown(
    """
<div class="banner">
  <div class="banner-title">🎬 Netflix Customer Churn Prediction Dashboard</div>
  <div class="banner-sub">🎬 Action! Subscriber engagement, retention patterns, and churn risk</div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="🎬 Rolling camera... downloading the dataset from Kaggle")
def load_from_kaggle():
    folder = kagglehub.dataset_download(KAGGLE_SLUG)
    csv_files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle folder.")
    return pd.read_csv(csv_files[0])


st.sidebar.title("🎬 Controls")
uploaded = st.sidebar.file_uploader("Or upload the CSV yourself", type="csv")

try:
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
    elif os.path.exists(LOCAL_CSV):
        raw_df = pd.read_csv(LOCAL_CSV)
    else:
        raw_df = load_from_kaggle()
except Exception as err:
    st.error(f"🎬 Cut! Could not load the dataset: {err}")
    st.info("Check KAGGLE_SLUG at the top of the file, or upload the CSV from the sidebar.")
    st.stop()

if "customer_id" in raw_df.columns:
    raw_df = raw_df.drop(columns=["customer_id"])
raw_df = raw_df.dropna().reset_index(drop=True)

if "churned" not in raw_df.columns:
    st.error("The dataset needs a 'churned' column (0/1).")
    st.stop()


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="🎬 Training the models...")
def train_models(data):
    X = data.drop(columns=["churned"])
    y = data["churned"]
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre_lr = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    lr = Pipeline([("pre", pre_lr), ("clf", LogisticRegression(max_iter=2000, random_state=42))])
    lr.fit(X_train, y_train)

    pre_rf = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    rf = Pipeline(
        [
            ("pre", pre_rf),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    rf.fit(X_train, y_train)

    lr_prob = lr.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    results = {
        "lr_acc": accuracy_score(y_test, lr.predict(X_test)),
        "rf_acc": accuracy_score(y_test, rf.predict(X_test)),
        "lr_auc": roc_auc_score(y_test, lr_prob),
        "rf_auc": roc_auc_score(y_test, rf_prob),
        "lr_roc": roc_curve(y_test, lr_prob),
        "rf_roc": roc_curve(y_test, rf_prob),
    }
    return lr, rf, results, num_cols, cat_cols


lr_model, rf_model, res, num_cols, cat_cols = train_models(raw_df)

# ---------------------------------------------------------------- sidebar filters
filtered = raw_df.copy()
if "subscription_type" in raw_df.columns:
    plans = st.sidebar.multiselect(
        "Subscription plan",
        sorted(raw_df["subscription_type"].unique()),
        default=sorted(raw_df["subscription_type"].unique()),
    )
    filtered = filtered[filtered["subscription_type"].isin(plans)]
if "region" in raw_df.columns:
    regions = st.sidebar.multiselect(
        "Region",
        sorted(raw_df["region"].unique()),
        default=sorted(raw_df["region"].unique()),
    )
    filtered = filtered[filtered["region"].isin(regions)]
st.sidebar.caption("🎬 Filters apply to Scene 1 only.")

if filtered.empty:
    st.warning("🎬 No subscribers match the current filters.")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    [
        "🎬 Scene 1 — Past: Explore",
        "🎬 Scene 2 — Present: Model Performance",
        "🎬 Scene 3 — Future: Churn Risk Predictor",
    ]
)

# ---------------------------------------------------------------- scene 1
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Subscribers", f"{len(filtered):,}", "kpi-choc")
    with k2:
        kpi("Churn rate", f"{filtered['churned'].mean() * 100:.1f}%", "kpi-gold")
    with k3:
        if "watch_hours" in filtered.columns:
            kpi("Avg watch hours", f"{filtered['watch_hours'].mean():.1f}", "kpi-cream")
    with k4:
        if "monthly_fee" in filtered.columns:
            kpi("Avg monthly fee", f"${filtered['monthly_fee'].mean():.2f}", "kpi-choc")

    st.write("")
    left, right = st.columns(2)

    with left:
        if "avg_watch_time_per_day" in filtered.columns:
            fig, ax = new_fig()
            stayed = filtered.loc[filtered["churned"] == 0, "avg_watch_time_per_day"]
            churned_vals = filtered.loc[filtered["churned"] == 1, "avg_watch_time_per_day"]
            ax.hist(stayed, bins=25, alpha=0.9, color=GOLD, label="Stayed")
            ax.hist(churned_vals, bins=25, alpha=0.8, color=CHOC, label="Churned")
            ax.set_title("Daily watch time: stayed vs churned")
            ax.set_xlabel("Hours per day")
            ax.legend()
            st.pyplot(fig)

    with right:
        if "subscription_type" in filtered.columns:
            fig, ax = new_fig()
            rate = filtered.groupby("subscription_type")["churned"].mean().sort_values()
            bar_colors = [CHOC if i % 2 == 0 else GOLD for i in range(len(rate))]
            ax.bar(rate.index.astype(str), rate.values * 100, color=bar_colors)
            ax.set_title("Churn rate by plan (%)")
            st.pyplot(fig)

    left2, right2 = st.columns(2)

    with left2:
        if "region" in filtered.columns:
            fig, ax = new_fig()
            rate = filtered.groupby("region")["churned"].mean().sort_values()
            ax.barh(rate.index.astype(str), rate.values * 100, color=GOLD, edgecolor=CHOC)
            ax.set_title("Churn rate by region (%)")
            st.pyplot(fig)

    with right2:
        if "last_login_days" in filtered.columns:
            fig, ax = new_fig()
            buckets = pd.cut(filtered["last_login_days"], bins=6)
            rate = filtered.groupby(buckets, observed=True)["churned"].mean() * 100
            ax.plot(range(len(rate)), rate.values, color=CHOC, marker="o", linewidth=2.5,
                    markerfacecolor=GOLD, markeredgecolor=CHOC, markersize=9)
            ax.set_xticks(range(len(rate)))
            ax.set_xticklabels([str(b) for b in rate.index], rotation=30, ha="right", fontsize=8)
            ax.set_title("Churn rate by days since last login (%)")
            st.pyplot(fig)

# ---------------------------------------------------------------- scene 2
with tab2:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi("Logistic Reg. accuracy", f"{res['lr_acc'] * 100:.1f}%", "kpi-cream")
    with m2:
        kpi("Logistic Reg. AUC", f"{res['lr_auc']:.3f}", "kpi-cream")
    with m3:
        kpi("Random Forest accuracy", f"{res['rf_acc'] * 100:.1f}%", "kpi-choc")
    with m4:
        kpi("Random Forest AUC", f"{res['rf_auc']:.3f}", "kpi-gold")

    st.write("")
    if res["rf_auc"] > res["lr_auc"]:
        st.info("🎬 Random Forest scores higher on AUC here, so it powers the predictor in Scene 3.")
    else:
        st.info("🎬 Logistic Regression scores at least as well on AUC here. The Random Forest still powers Scene 3.")

    a, b = st.columns(2)

    with a:
        fig, ax = new_fig()
        fpr, tpr, _ = res["lr_roc"]
        ax.plot(fpr, tpr, color=GOLD, linewidth=2.5, label=f"Logistic ({res['lr_auc']:.3f})")
        fpr, tpr, _ = res["rf_roc"]
        ax.plot(fpr, tpr, color=CHOC, linewidth=2.5, label=f"Random Forest ({res['rf_auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
        ax.set_title("ROC curves")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        st.pyplot(fig)

    with b:
        names = rf_model.named_steps["pre"].get_feature_names_out()
        names = [n.replace("num__", "").replace("cat__", "") for n in names]
        importance = pd.Series(rf_model.named_steps["clf"].feature_importances_, index=names)
        top = importance.sort_values().tail(10)
        fig, ax = new_fig()
        ax.barh(top.index, top.values, color=[CHOC if i % 2 == 0 else GOLD for i in range(len(top))])
        ax.set_title("Top 10 churn drivers (Random Forest)")
        st.pyplot(fig)

# ---------------------------------------------------------------- scene 3
with tab3:
    st.subheader("🎬 Subscriber churn risk calculator")
    feature_cols = num_cols + cat_cols
    values = {}
    col_a, col_b = st.columns(2)

    for idx, name in enumerate(feature_cols):
        target = col_a if idx % 2 == 0 else col_b
        label = name.replace("_", " ").title()
        with target:
            if name in num_cols:
                lo = float(raw_df[name].min())
                hi = float(raw_df[name].max())
                mid = float(raw_df[name].median())
                if pd.api.types.is_integer_dtype(raw_df[name]):
                    values[name] = st.slider(label, int(lo), int(hi), int(mid))
                else:
                    values[name] = st.slider(label, lo, hi, mid)
            else:
                values[name] = st.selectbox(label, sorted(raw_df[name].unique()))

    if st.button("🎬 Action! Assess churn risk"):
        input_df = pd.DataFrame([values])[feature_cols]
        risk = rf_model.predict_proba(input_df)[0][1]
        st.progress(min(max(float(risk), 0.0), 1.0))
        if risk > 0.5:
            st.error(f"🚨 High churn risk: {risk * 100:.0f}%. Consider a retention offer or a re-engagement nudge.")
        else:
            st.success(f"✅ Low churn risk: {risk * 100:.0f}%. This subscriber looks stable.")