# Customer Attrition Predictor

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation and Objectives](#motivation-and-objectives)
- [Data Source](#data-source)
- [Tools](#tools)
- [Project Workflow](#project-workflow)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Data Visualization](#data-visualization)
- [Insights and Next Steps](#insights-and-next-steps)
  - [Key Findings](#key-findings)
  - [Next Steps](#next-steps)
- [Limitations](#limitations)
- [Impact and Future Plans](#impact-and-future-plans)
- [References](#references)
- [Contact Information](#contact-information)

## Project Overview
The goal of this project is to develop a **supervised machine learning model** to predict **customer attrition (churn)** in a banking environment. The model leverages historical customer data to forecast the likelihood of a client leaving the bank, enabling proactive retention strategies. **XGBoost** was used as the primary model due to its strong performance on tabular data.

## Motivation and Objectives
Customer churn is costly for banks. Predicting which customers are at risk allows banks to **retain high-value clients, improve customer experience, and optimize marketing strategies**.  

### Objectives
- Analyze historical customer data to uncover patterns leading to churn.
- Build a predictive model to forecast customer attrition.
- Identify the **most important features** driving churn using feature importance analysis.
- Evaluate model performance using metrics such as **ROC-AUC** and **classification reports**.
- Provide actionable next steps for retention strategies.

## Data Source
The dataset includes historical banking customer data such as demographic information, account details, credit behavior, transaction activity, and engagement metrics.  
Data was preprocessed for modeling purposes and can be sourced from internal banking records or publicly available churn datasets.

## Tools
**Python** for coding:
- `Pandas` for data manipulation
- `NumPy` for numerical operations
- `Scikit-learn` for preprocessing, model evaluation, and feature selection
- `XGBoost` for supervised machine learning
- `Matplotlib` / `Seaborn` for visualization
- **Jupyter Notebook** / **Streamlit** for interactive development and app deployment

## Project Workflow

### Data Cleaning
- Handled missing values and corrected inconsistent data entries.
- Encoded categorical variables for model compatibility.
- Scaled numeric features using `StandardScaler` for consistent modeling.

### Exploratory Data Analysis
- Examined distributions of key features such as **tenure, transaction counts, credit utilization, and account balances**.
- Identified trends between churned and existing customers to guide feature selection.

### Data Visualization
- Created boxplots, histograms, and bar charts to visualize patterns.
- Highlighted features with significant differences between churned and existing customers.
- Correlation heatmaps were used to identify relationships among numeric features.

## Insights and Next Steps

### Key Findings
- Customers with **shorter tenure**, **fewer products**, and **high inactivity** were more likely to churn.
- High **credit utilization** combined with lower credit limits indicated financial stress, leading to higher churn probability.
- Drops in **transaction count** and **spending** were strong predictors of churn.
- Demographic features such as **age** and **number of dependents** had minimal influence on churn.

### Next Steps
1. **Monitor and Retrain Model Regularly**  
   - Update the model with new customer data to maintain accuracy over time.
2. **Deploy Real-Time Alerts**  
   - Integrate predictions into banking dashboards to flag at-risk customers early.
3. **Feature Expansion**  
   - Incorporate additional engagement, product usage, and behavioral metrics for better accuracy.
4. **Experiment with Ensemble Models**  
   - Test other algorithms or ensembles for improved performance.
5. **Business Integration**  
   - Use insights to design targeted retention campaigns for customers with high churn probability.

## Limitations
- Limited by historical data; may not capture future behavior changes.
- Model performance depends on data quality and completeness.
- Some features (demographics) have low predictive power and may require additional behavioral data for improved predictions.

## Impact and Future Plans
The model provides actionable insights for **reducing customer attrition** and optimizing banking strategies. Future enhancements include:
- Integrating the model into a **Streamlit app** for easy use by business teams.
- Continuously **monitoring model performance** and updating with new data.
- Expanding the dataset with **external economic or behavioral indicators** to strengthen predictions.

  ### Try the Customer Attrition Predictor App

You can access the interactive Streamlit app here:  
[Customer Churn Predictor](https://customerattritionpredictorurl.streamlit.app/)


## References
- Kaggle customer churn datasets and related tutorials
- XGBoost and Scikit-learn official documentation
- Articles and tutorials on customer retention analytics

## Contact Information
For questions or collaboration:
- **Name:** Satelite Alison Ndayikunda
- **Email:** [satalisonn@gmail.com](mailto:satalisonn@gmail.com)
