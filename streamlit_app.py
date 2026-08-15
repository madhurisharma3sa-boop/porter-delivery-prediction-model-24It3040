import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Porter Delivery Time Predictor",
    page_icon="🛵",
    layout="wide"
)


NUMERIC_FEATURES = [
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_partners",
    "total_busy_partners",
    "total_outstanding_orders",
    "order_hour",
    "order_day_of_week",
    "order_is_weekend",
    "hour_sin",
    "hour_cos",
    "price_range",
    "avg_item_price",
    "items_per_distinct",
    "busy_partner_ratio",
    "available_partners",
    "outstanding_per_partner"
]

CATEGORICAL_FEATURES = [
    "market_id",
    "order_protocol",
    "store_primary_category"
]


@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_columns():
    with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_category_reference():
    with open(os.path.join(BASE_DIR, "category_reference.pkl"), "rb") as f:
        return pickle.load(f)


def engineer_input(raw, category_reference, feature_columns):

    df = pd.DataFrame([raw])

    # Time features
    df["order_hour"] = df["order_hour"].astype(int)
    df["order_day_of_week"] = df["order_day_of_week"].astype(int)

    df["order_is_weekend"] = (
        df["order_day_of_week"].isin([5, 6]).astype(int)
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["order_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["order_hour"] / 24
    )

    # Price features
    df["price_range"] = (
        df["max_item_price"] - df["min_item_price"]
    )

    df["avg_item_price"] = (
        df["subtotal"] /
        df["total_items"].replace(0, 1)
    )

    df["items_per_distinct"] = (
        df["total_items"] /
        df["num_distinct_items"].replace(0, 1)
    )

    # Marketplace features
    df["busy_partner_ratio"] = (
        df["total_busy_partners"] /
        (df["total_onshift_partners"] + 1)
    )

    df["available_partners"] = (
        df["total_onshift_partners"] -
        df["total_busy_partners"]
    ).clip(lower=0)

    df["outstanding_per_partner"] = (
        df["total_outstanding_orders"] /
        (df["total_onshift_partners"] + 1)
    )

    # Handle unseen store category
    if (
        df.loc[0, "store_primary_category"]
        not in category_reference["store_primary_category"]
    ):
        if "other" in category_reference["store_primary_category"]:
            df["store_primary_category"] = "other"
        else:
            df["store_primary_category"] = (
                category_reference["store_primary_category"][0]
            )

    # Select and encode features
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES
    )

    # Match training columns exactly
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X


def predict_delivery_time(raw):

    model = load_model()
    scaler = load_scaler()
    feature_columns = load_feature_columns()
    category_reference = load_category_reference()

    X = engineer_input(
        raw,
        category_reference,
        feature_columns
    )

    X_scaled = scaler.transform(X)

    prediction = model.predict(
        X_scaled,
        verbose=0
    ).flatten()[0]

    return max(float(prediction), 1.0)


# Check required files
REQUIRED_FILES = [
    "model.pkl",
    "scaler.pkl",
    "feature_columns.pkl",
    "category_reference.pkl"
]

missing_files = [
    file for file in REQUIRED_FILES
    if not os.path.exists(os.path.join(BASE_DIR, file))
]

if missing_files:
    st.error("Required model files are missing.")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


category_reference = load_category_reference()


# =========================
# UI
# =========================

st.title("🛵 Porter Delivery Time Predictor")

st.caption(
    "Predict delivery time using order details and marketplace conditions."
)

col1, col2, col3 = st.columns(3)


# Order Details
with col1:

    st.subheader("Order Details")

    total_items = st.number_input(
        "Total items",
        min_value=1,
        max_value=50,
        value=3
    )

    num_distinct_items = st.number_input(
        "Distinct items",
        min_value=1,
        max_value=50,
        value=2
    )

    subtotal = st.number_input(
        "Subtotal (cents)",
        min_value=0,
        max_value=20000,
        value=2500,
        step=100
    )

    min_item_price = st.number_input(
        "Min item price (cents)",
        min_value=0,
        max_value=10000,
        value=500,
        step=50
    )

    max_item_price = st.number_input(
        "Max item price (cents)",
        min_value=0,
        max_value=10000,
        value=1500,
        step=50
    )


# Store & Order Context
with col2:

    st.subheader("Store & Order Context")

    store_primary_category = st.selectbox(
        "Store category",
        category_reference["store_primary_category"]
    )

    market_id = st.selectbox(
        "Market ID",
        category_reference["market_id"]
    )

    order_protocol = st.selectbox(
        "Order protocol",
        category_reference["order_protocol"]
    )

    order_hour = st.slider(
        "Order hour",
        0,
        23,
        19
    )

    order_day_of_week = st.selectbox(
        "Day of week",
        range(7),
        format_func=lambda x: [
            "Mon", "Tue", "Wed",
            "Thu", "Fri", "Sat", "Sun"
        ][x],
        index=4
    )


# Marketplace Conditions
with col3:

    st.subheader("Marketplace Conditions")

    total_onshift_partners = st.number_input(
        "Partners on shift",
        min_value=0,
        max_value=200,
        value=20
    )

    total_busy_partners = st.number_input(
        "Busy partners",
        min_value=0,
        max_value=200,
        value=15
    )

    total_outstanding_orders = st.number_input(
        "Outstanding orders",
        min_value=0,
        max_value=300,
        value=25
    )


if st.button(
    "🔮 Predict Delivery Time",
    type="primary",
    use_container_width=True
):

    raw_input = {
        "total_items": total_items,
        "subtotal": subtotal,
        "num_distinct_items": num_distinct_items,
        "min_item_price": min_item_price,
        "max_item_price": max_item_price,
        "total_onshift_partners": total_onshift_partners,
        "total_busy_partners": total_busy_partners,
        "total_outstanding_orders": total_outstanding_orders,
        "order_hour": order_hour,
        "order_day_of_week": order_day_of_week,
        "market_id": market_id,
        "order_protocol": order_protocol,
        "store_primary_category": store_primary_category
    }

    with st.spinner("Predicting delivery time..."):
        prediction = predict_delivery_time(raw_input)

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Estimated Delivery Time",
            f"{prediction:.1f} min"
        )

        st.caption(
            f"Typical range: "
            f"{max(prediction - 8, 1):.0f}–"
            f"{prediction + 8:.0f} min"
        )

    with col2:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Delivery ETA (minutes)"},
                gauge={
                    "axis": {"range": [0, 120]},
                    "bar": {"color": "#FF6B35"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4f7d4"},
                        {"range": [30, 60], "color": "#fff2cc"},
                        {"range": [60, 120], "color": "#f8cccc"}
                    ]
                }
            )
        )

        fig.update_layout(
            height=280,
            margin=dict(t=40, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
