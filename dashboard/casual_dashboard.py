#!/usr/bin/env python
# coding: utf-8

import os #lets you handle file paths
import streamlit as st #Streamlit creates web apps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor#ML model that predicts outcomes
from sklearn.preprocessing import StandardScaler #standardizes numeric data
from sklearn.metrics import mean_squared_error, r2_score #metrics to measure how well models perform.
from sklearn.exceptions import NotFittedError #used if model isn’t trained yet
from sklearn.linear_model import LogisticRegression #predicts probability of treatment
from sklearn.neighbors import NearestNeighbors #helps find “similar” individuals between treatment and control groups.

st.set_page_config(page_title="Causal Inference Dashboard", layout="wide")


@st.cache_data #remember the result
def load_data():
    try:
        base_path = os.path.dirname(os.path.dirname(__file__))  # go up from dashboard/
        dataset1 = os.path.join(base_path, "datasets", "balanced_ihdata.csv")
        dataset2 = os.path.join(base_path, "datasets", "ihdata.csv")

        if os.path.exists(dataset1):
            data = pd.read_csv(dataset1)
        elif os.path.exists(dataset2):
            data = pd.read_csv(dataset2)
        else:
            st.warning("Neither balanced_ihdata.csv nor ihdata.csv found. Using synthetic data.")
            return create_synthetic_data() #If no dataset file exists, it creates synthetic (fake) data

        data = data.loc[:, ~data.columns.str.contains("^Unnamed")] #removes unnamed columns
        data["treatment"] = data["treatment"].astype(float)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_synthetic_data()

def create_synthetic_data(n=500, random_seed=42): #fake data with 500 people
    np.random.seed(random_seed)
    data = pd.DataFrame({
        "treatment": np.concatenate([np.ones(n//2), np.zeros(n//2)]),
        "income": np.random.normal(50000, 10000, n),
        "birth_weight": np.random.normal(3000, 500, n),
        "parent_edu": np.random.randint(8, 18, n),
        "health_index": np.random.normal(70, 15, n),
        "housing_quality": np.random.normal(6, 2, n),
        "neighborhood_safety": np.random.normal(7, 2, n)
    })
    
    beta = np.array([0.5, 0.3, 0.8, 0.4, 0.6, 0.2]) #what would happen if treated vs not
    confounders = ["income", "birth_weight", "parent_edu", "health_index", "housing_quality", "neighborhood_safety"]
    X = data[confounders].values
    data["mu0"] = X.dot(beta) + np.random.normal(0, 1, n)
    data["mu1"] = data["mu0"] + 5 + 0.1 * data["income"]/10000 + 0.2 * data["parent_edu"]
    
    data["outcome_factual"] = np.where(data["treatment"] == 1, data["mu1"], data["mu0"]) #mu=outcome if not treated, m1=outcome if treated
    data["outcome_counterfactual"] = np.where(data["treatment"] == 1, data["mu0"], data["mu1"])
    
    for i in range(11, 26):
        data[f"x{i}"] = np.random.normal(0, 1, n)
    return data


if 'data' not in st.session_state or st.sidebar.button("Reset Data"): #storing data in session
    st.session_state.data = load_data()
    st.session_state.models_fitted = False

data = st.session_state.data
confounders = ["income", "birth_weight", "parent_edu", "health_index", "housing_quality", "neighborhood_safety"]

if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler().fit(data[confounders])

#MODEL TRAINING
@st.cache_data(show_spinner=True)
def train_model(data, treatment_group, confounders):
    subset = data[data["treatment"] == treatment_group]#looking at the data relevant to the treatment group we want to model.
    if len(subset) == 0: #Checks if the filtered subset is empty
        raise ValueError(f"No data for treatment group {treatment_group}")
    X, y = subset[confounders], subset["outcome_factual"] #x=features, y=target
    X_scaled = st.session_state.scaler.transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred) #how well the model predicts outcome.
    mse = mean_squared_error(y, y_pred)
    return model, r2, mse

def predict_outcome(model, input_data):
    try:
        input_data_scaled = st.session_state.scaler.transform(input_data)
        return model.predict(input_data_scaled)[0]
    except NotFittedError:
        st.error("Model not fitted yet.")
        return None


page = st.sidebar.radio("Navigate", ["Input & Results", "Dataset Overview", "Model Diagnostics"])

n_treated = (data["treatment"] == 1).sum()
n_control = (data["treatment"] == 0).sum()
st.sidebar.write(f"Dataset: {len(data)} samples")
st.sidebar.write(f"- Treated group: {n_treated}")
st.sidebar.write(f"- Control group: {n_control}")

try:
    if 'models_fitted' not in st.session_state or not st.session_state.models_fitted:
        with st.spinner("Training models..."):
            st.session_state.treated_model, st.session_state.treated_r2, st.session_state.treated_mse = train_model(data, 1, confounders)
            st.session_state.control_model, st.session_state.control_r2, st.session_state.control_mse = train_model(data, 0, confounders)
            st.session_state.models_fitted = True
except Exception as e:
    st.error(f"Error training models: {str(e)}")
    st.session_state.models_fitted = False


if page == "Input & Results":
    st.title("Causal Inference Analysis")

    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Income", float(data["income"].min()), float(data["income"].max()), float(data["income"].median()))
        birth_weight = st.slider("Birth Weight", float(data["birth_weight"].min()), float(data["birth_weight"].max()), float(data["birth_weight"].median()))
        parent_edu = st.slider("Parent Education", int(data["parent_edu"].min()), int(data["parent_edu"].max()), int(data["parent_edu"].median()))
    with col2:
        health_index = st.slider("Health Index", float(data["health_index"].min()), float(data["health_index"].max()), float(data["health_index"].median()))
        housing_quality = st.slider("Housing Quality", float(data["housing_quality"].min()), float(data["housing_quality"].max()), float(data["housing_quality"].median()))
        neighborhood_safety = st.slider("Neighborhood Safety", float(data["neighborhood_safety"].min()), float(data["neighborhood_safety"].max()), float(data["neighborhood_safety"].median()))

    treatment = st.radio("Treatment Applied?", ["Yes", "No"])
    treatment_value = 1 if treatment == "Yes" else 0

    if st.session_state.models_fitted:
        input_data = pd.DataFrame({
            "income": [income],
            "birth_weight": [birth_weight],
            "parent_edu": [parent_edu],
            "health_index": [health_index],
            "housing_quality": [housing_quality],
            "neighborhood_safety": [neighborhood_safety]
        })

        treated_outcome = predict_outcome(st.session_state.treated_model, input_data)
        control_outcome = predict_outcome(st.session_state.control_model, input_data)

        if treated_outcome is not None and control_outcome is not None:
            individual_treatment_effect = treated_outcome - control_outcome
            factual_outcome = treated_outcome if treatment_value else control_outcome
            counterfactual_outcome = control_outcome if treatment_value else treated_outcome

            st.subheader("Individual Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Factual Outcome", f"{factual_outcome:.2f}")
            with col2:
                st.metric("Counterfactual Outcome", f"{counterfactual_outcome:.2f}")
            st.metric("Individual Treatment Effect (ITE)", f"{individual_treatment_effect:.2f}")

            # --- POPULATION LEVEL RESULTS ---
            st.subheader("Population Results")

            # Average Treatment Effect (ATE)
            treated_mean = data[data["treatment"] == 1]["outcome_factual"].mean()
            control_mean = data[data["treatment"] == 0]["outcome_factual"].mean()
            ate = treated_mean - control_mean
            st.metric("Average Treatment Effect (ATE)", f"{ate:.2f}")

            # Propensity Score Matching (PSM ATE)
            log_reg = LogisticRegression()
            X_psm = data[confounders]
            y_psm = data["treatment"]
            log_reg.fit(X_psm, y_psm)
            data["propensity_score"] = log_reg.predict_proba(X_psm)[:, 1]

            treated_df = data[data["treatment"] == 1]
            control_df = data[data["treatment"] == 0]

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(control_df[["propensity_score"]])
            distances, indices = nn.kneighbors(treated_df[["propensity_score"]])
            matched_controls = control_df.iloc[indices.flatten()]

            psm_ate = (treated_df["outcome_factual"].values - matched_controls["outcome_factual"].values).mean()
            st.metric("Propensity Score Matching ATE", f"{psm_ate:.2f}")

# ------------------ DATASET OVERVIEW ------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")

    st.dataframe(data.head())

    st.subheader("Treatment Group Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=["Control", "Treated"], y=[n_control, n_treated], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Number of Samples by Treatment Group")
    st.pyplot(fig)

    st.subheader("Covariate Balance")
    balance_stats = []
    for var in confounders:
        treated_mean = data[data["treatment"] == 1][var].mean()
        control_mean = data[data["treatment"] == 0][var].mean()
        std_diff = (treated_mean - control_mean) / data[var].std()
        balance_stats.append({
            "Variable": var,
            "Treated Mean": treated_mean,
            "Control Mean": control_mean,
            "Diff": treated_mean - control_mean,
            "Std Diff": std_diff,
            "Balanced": abs(std_diff) < 0.25
        })
    st.dataframe(pd.DataFrame(balance_stats))

    st.subheader("Confounder Distributions by Treatment Group")
    confounder_col1, confounder_col2 = st.columns(2)
    for i, confounder in enumerate(confounders):
        col = confounder_col1 if i % 2 == 0 else confounder_col2
        with col:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=data, x=confounder, hue="treatment",
                         common_norm=False, element="step",
                         palette=["blue", "red"], ax=ax)
            ax.set_title(f"{confounder} Distribution by Treatment Group")
            st.pyplot(fig)

    st.subheader("Distribution of Outcomes for Treated vs Control")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[data['treatment'] == 1]['outcome_factual'],
                 color='blue', label='Treated', kde=True, ax=ax)
    sns.histplot(data[data['treatment'] == 0]['outcome_factual'],
                 color='red', label='Control', kde=True, ax=ax)
    ax.set_title('Distribution of Outcomes for Treated and Control Groups')
    ax.set_xlabel('Outcome (outcome_factual)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Boxplot of Outcomes by Treatment Group")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='treatment', y='outcome_factual', data=data, ax=ax)
    ax.set_title('Boxplot of Outcomes by Treatment Group')
    ax.set_xlabel('Treatment')
    ax.set_ylabel('Outcome (outcome_factual)')
    st.pyplot(fig)

# ------------------ MODEL DIAGNOSTICS ------------------
elif page == "Model Diagnostics":
    st.title("Model Diagnostics")
    if st.session_state.models_fitted:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Treated Model")
            st.metric("R²", f"{st.session_state.treated_r2:.4f}")
            st.metric("MSE", f"{st.session_state.treated_mse:.4f}")
        with col2:
            st.subheader("Control Model")
            st.metric("R²", f"{st.session_state.control_r2:.4f}")
            st.metric("MSE", f"{st.session_state.control_mse:.4f}")
    else:
        st.warning("Models not fitted yet. Please reload data.")
