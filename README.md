# Causal Inference on Infant Health Dataset

Exploring causal relationships in infant health data using advanced modelling techniques and interactive dashboards.

---

## Project Overview

This project investigates **causal effects and relationships** in infant health metrics using a publicly available dataset.  
The objective is to move **beyond correlation** and uncover meaningful **causal insights**, such as how prenatal and postnatal factors influence infant health outcomes.

---

## Key Highlights

- Data preprocessing and cleaning  
- Causal modelling using **Propensity Scores** and **Inverse Probability Weighting (IPW)**  
- Interactive dashboards for exploratory and causal analysis  
- Transparent and reproducible analysis using **Jupyter Notebooks**

---

## Repository Structure

sml_project/
│
├── datasets/                   # Raw and cleaned infant health datasets
│   ├── balanced_ihdata.csv
│   ├── unbalanced_ihdata.csv
│
├── dashboard/                  # Streamlit dashboard application
│   ├── dashboard.py
│   └──                  # Plots / UI resources (if any)
│
├── Casual_Model.ipynb          # Causal inference modelling notebook
├── IPW.ipynb                   # Inverse Probability Weighting analysis
└── .gitignore                  # Ignored files and folders


---

## Methods & Approach

### 1. Data Exploration
- Visualized data distributions, missing values, and feature relationships
- Identified key variables affecting infant health outcomes

### 2. Causal Framework Setup
- Defined **treatment, outcome, and confounder variables**
- Established causal assumptions and study design

### 3. Model Implementation
- Propensity score estimation  
- Inverse Probability Weighting (IPW)  
- Sensitivity and robustness analysis

### 4. Dashboarding & Visualization
- Built interactive dashboards to explore results
- Summarized causal effects using visual analytics

### 5. Interpretation & Reporting
- Interpreted causal estimates in the context of infant health
- Derived insights to support data-driven conclusions

---

## Tools & Technologies

- Python  
- Pandas, NumPy, Scikit-learn  
- Jupyter Notebook  
- Streamlit  

