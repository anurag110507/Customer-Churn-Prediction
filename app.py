import os

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Retainova | Customer Churn",
    page_icon="📊",
    layout="wide"
)

st.title("Retainova")
st.subheader("AI-Powered Customer Churn Prediction")
st.caption("Understand churn patterns and identify customers who may leave.")


# -----------------------------
# Load data and model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "customer_churn_model.pkl")


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    ).fillna(0)
    return data


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.info("Save your trained model from Model_Training.ipynb first.")
    st.stop()

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)


# -----------------------------
# Overview metrics
# -----------------------------
total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
churn_rate = (
    churned_customers / total_customers * 100
    if total_customers else 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.2f}%")
col4.metric(
    "Retained Customers",
    f"{total_customers - churned_customers:,}"
)

st.divider()


# -----------------------------
# Navigation
# -----------------------------
overview_tab, prediction_tab, risk_tab = st.tabs([
    "📊 Churn Analytics",
    "🔮 Customer Prediction",
    "🛡️ Customer Risk Analysis"
])


# -----------------------------
# Churn Analytics tab
# -----------------------------
with overview_tab:
    st.header("Customer Churn Analytics")

    left, right = st.columns(2)

    with left:
        st.subheader("Churn Distribution")
        counts = df["Churn"].value_counts().reindex(
            ["No", "Yes"], fill_value=0
        )
        st.bar_chart(counts)

    with right:
        st.subheader("Churn by Contract")
        contract_table = pd.crosstab(
            df["Contract"], df["Churn"]
        ).reindex(columns=["No", "Yes"], fill_value=0)
        st.bar_chart(contract_table)

    st.subheader("Monthly Charges vs Churn")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly Charges by Churn Status")
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# Individual customer prediction tab
# -----------------------------
with prediction_tab:
    st.header("Predict Customer Churn")
    st.write(
        "Enter a customer's information to estimate their churn risk."
    )

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=2000.0,
                value=70.0
            )

        col1, col2 = st.columns(2)

        with col1:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn", type="primary"
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }])

        try:
            probability = model.predict_proba(customer)[0][1]
            prediction = int(probability >= 0.5)

            st.divider()
            st.subheader("Prediction Result")
            st.metric(
                "Estimated Churn Probability",
                f"{probability:.1%}"
            )

            if prediction == 1:
                st.warning(
                    "Higher predicted churn risk. Consider reviewing "
                    "this customer's needs and retention options."
                )
            else:
                st.success(
                    "Lower predicted churn risk at the current "
                    "classification threshold."
                )

            st.caption(
                "This is a model estimate, not a guarantee that "
                "a customer will or will not leave."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the saved model was trained with the "
                "same dataset columns and preprocessing pipeline."
            )


# -----------------------------
# Customer Risk Analysis tab
# -----------------------------
with risk_tab:
    st.header("Customer Risk Analysis")
    st.write(
        "Review customers ranked by the churn probability estimated "
        "by the trained model."
    )

    risk_features = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    ).copy()

    risk_features["TotalCharges"] = pd.to_numeric(
        risk_features["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    try:
        probabilities = model.predict_proba(risk_features)[:, 1]
    except Exception as exc:
        st.error(f"Could not calculate customer risk scores: {exc}")
        st.stop()

    risk_df = pd.DataFrame({
        "Customer ID": df["customerID"],
        "Contract": df["Contract"],
        "Internet Service": df["InternetService"],
        "Tenure (Months)": df["tenure"],
        "Monthly Charges": df["MonthlyCharges"],
        "Actual Churn": df["Churn"],
        "Churn Probability": probabilities
    })

    # These are initial business-rule categories; tune them after validation.
    risk_df["Risk Level"] = risk_df["Churn Probability"].apply(
        lambda p: (
            "High Risk" if p >= 0.60
            else "Medium Risk" if p >= 0.30
            else "Low Risk"
        )
    )

    high_risk = (risk_df["Risk Level"] == "High Risk").sum()
    medium_risk = (risk_df["Risk Level"] == "Medium Risk").sum()
    low_risk = (risk_df["Risk Level"] == "Low Risk").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", f"{high_risk:,}")
    col2.metric("Medium Risk", f"{medium_risk:,}")
    col3.metric("Low Risk", f"{low_risk:,}")

    st.caption(
        "This page scores customers already present in the historical "
        "dataset. Actual Churn is the recorded outcome; Churn Probability "
        "is the model's estimate. These results alone do not measure "
        "performance on unseen customers."
    )

    st.divider()
    st.subheader("Filter Customers")

    col1, col2 = st.columns(2)

    with col1:
        selected_risk = st.multiselect(
            "Filter by Risk Level",
            ["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"]
        )

    with col2:
        contracts = sorted(risk_df["Contract"].dropna().unique().tolist())
        selected_contract = st.multiselect(
            "Filter by Contract",
            contracts,
            default=contracts
        )

    filtered_df = risk_df[
        risk_df["Risk Level"].isin(selected_risk)
        & risk_df["Contract"].isin(selected_contract)
    ].copy()

    filtered_df = filtered_df.sort_values(
        by="Churn Probability",
        ascending=False
    )

    display_df = filtered_df.copy()
    display_df["Churn Probability"] = (
        display_df["Churn Probability"] * 100
    ).round(2)

    st.subheader("Customers Ranked by Churn Risk")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Risk Analysis CSV",
        data=csv_data,
        file_name="customer_risk_analysis.csv",
        mime="text/csv"
    )

    # -----------------------------
    # Retention Recommendations
    # -----------------------------
    st.divider()
    st.subheader("Retention Recommendations")
    st.write(
        "These are rule-based suggestions to help plan retention actions. "
        "Review them with customer context before taking action."
    )

    def get_recommendation(row):
        recommendations = []

        if row["Contract"] == "Month-to-month":
            recommendations.append(
                "Offer an incentive to consider a longer-term contract"
            )

        if row["Internet Service"] == "Fiber optic":
            recommendations.append(
                "Review internet service quality and customer satisfaction"
            )

        if row["Monthly Charges"] > 80:
            recommendations.append(
                "Review plan value and consider an appropriate loyalty offer"
            )

        if row["Tenure (Months)"] < 12:
            recommendations.append(
                "Offer onboarding support and check early customer experience"
            )

        if not recommendations:
            recommendations.append(
                "Conduct a satisfaction check-in and offer personalized support"
            )

        return " | ".join(recommendations)

    recommendations_df = risk_df.copy()
    recommendations_df["Recommended Action"] = recommendations_df.apply(
        get_recommendation,
        axis=1
    )

    high_risk_customers = recommendations_df[
        recommendations_df["Risk Level"] == "High Risk"
    ].sort_values(
        by="Churn Probability",
        ascending=False
    ).copy()

    if high_risk_customers.empty:
        st.info(
            "No high-risk customers were identified using the current "
            "risk-category rules."
        )
    else:
        high_risk_display = high_risk_customers[
            [
                "Customer ID",
                "Churn Probability",
                "Contract",
                "Monthly Charges",
                "Recommended Action"
            ]
        ].copy()

        high_risk_display["Churn Probability"] = (
            high_risk_display["Churn Probability"] * 100
        ).round(2)

        st.dataframe(
            high_risk_display,
            use_container_width=True,
            hide_index=True
        )

        recommendation_csv = high_risk_display.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download High-Risk Retention Actions",
            data=recommendation_csv,
            file_name="high_risk_retention_actions.csv",
            mime="text/csv"
        )
