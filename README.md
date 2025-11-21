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
  - [Model Selection and Training](#model-selection-and-training)
- [Insights and Next Steps](#insights-and-next-steps)
  - [Key Findings](#key-findings)
  - [Next Steps](#next-steps)
- [Limitations](#limitations)
- [Impact and Future Plans](#impact-and-future-plans)
- [References](#references)
- [Contact Information](#contact-information)

---

## Project Overview
The **Customer Attrition Predictor** is a supervised machine learning project that forecasts the likelihood of a banking customer **churning** (leaving the bank).  
The project leverages historical customer data, including demographics, account usage, credit behavior, and transaction history, to predict churn and highlight actionable insights for retention.  

The primary model used is **XGBoost**, chosen for its ability to handle **non-linear relationships** and deliver **high predictive performance** on tabular data.

---

## Motivation and Objectives
Customer churn is a significant challenge in the banking industry because losing clients directly impacts revenue and profitability. Predicting churn allows banks to:

- Identify at-risk customers early.
- Implement proactive retention strategies.
- Optimize marketing and loyalty campaigns.
- Focus resources on high-value clients.

### Objectives
- Explore customer data to detect trends and patterns related to churn.
- Develop a **predictive model** using supervised learning techniques.
- Evaluate model performance using metrics like **ROC-AUC, F1-score, and confusion matrices**.
- Identify **key features** driving churn for actionable business insights.
- Build a **Streamlit app** for interactive prediction and risk visualization.

---

## Data Source
The dataset includes historical records of bank customers with features such as:

- Demographics: **age, gender, number of dependents**
- Account & credit: **credit limit, revolving balance, utilization ratio**
- Engagement & activity: **transaction counts, amount spent, inactivity months**
- Relationship metrics: **number of products held, tenure with bank**

> Note: Data may be anonymized or sourced from public churn datasets for demonstration purposes.  

---

## Tools
- **Python** for coding and analysis.
- **Pandas** and **NumPy** for data manipulation.
- **Scikit-learn** for preprocessing, modeling, and evaluation.
- **XGBoost** for supervised classification.
- **Matplotlib** and **Seaborn** for visualization.
- **Jupyter Notebook** for interactive development.
- **Streamlit** for deploying the prediction tool.

---

## Project Workflow

### Data Cleaning
- Handled missing values and inconsistencies in numeric and categorical columns.
- Encoded categorical variables (e.g., gender, marital status, education, income category) for model compatibility.
- Scaled numeric features using `StandardScaler` to improve model performance and support distance-based methods like SMOTE.

### Exploratory Data Analysis
- Examined distributions of numeric features like **credit limit, total transaction amount, utilization ratio**, and **months on book**.
- Identified relationships between churn and categorical variables such as **gender, marital status, and education level**.
- Visualized correlations between numeric features to detect redundancy or multicollinearity.
  
**Example visualizations you can include:**
- Histograms and boxplots for numeric features (e.g., `Credit_Limit`, `Total_Trans_Amt`)  
![Histogram Example](path_to_histogram_image.png)
- Bar charts for categorical features (e.g., `Education_Level`, `Marital_Status`)  
![Categorical Bar Chart](path_to_bar_chart_image.png)
- Correlation heatmap of numeric features  
![Correlation Heatmap](path_to_correlation_heatmap_image.png)

---

### Data Visualization
- Plotted **distributions of churned vs existing customers** for key features.
- Highlighted differences in transaction behavior, credit utilization, and tenure to identify predictors of churn.
- Annotated visualizations with percentages and medians for easy interpretation.

---

### Model Selection and Training
- Evaluated multiple supervised models:
  - **Logistic Regression**: Baseline linear model for churn prediction.
  - **Random Forest**: Captures non-linear relationships and provides feature importance.
  - **Gradient Boosting**: Boosted ensemble for improved predictive accuracy.
  - **XGBoost**: Selected as final model due to **highest ROC-AUC and F1-score**.
- Used **SMOTE** for handling class imbalance in training data.
- Performed **RFECV (Recursive Feature Elimination with Cross-Validation)** to select the most predictive features.

**Evaluation Metrics**
- Confusion Matrix  
![Confusion Matrix](path_to_confusion_matrix_image.png)
- ROC-AUC Score
- Precision, Recall, F1-score

**Key Advantages of XGBoost**
- Handles missing values natively.
- Robust to outliers and multicollinearity.
- Provides feature importance for actionable insights.

---

## Insights and Next Steps

### Key Findings
- **Shorter tenure** and **fewer products** → higher churn risk.
- **High inactivity** and **high credit utilization** → strong indicators of churn.
- **Drops in transaction amount and count** often precede attrition.
- Demographics like **age** and **dependents** have limited predictive power.

### Next Steps
1. **Regular Model Retraining**  
   - Keep the model updated with new data for reliable predictions.
2. **Real-Time Alerts**  
   - Flag high-risk customers for targeted retention campaigns.
3. **Feature Expansion**  
   - Include additional behavioral and product usage metrics.
4. **Experiment with Other Models**  
   - Explore ensemble or neural network models to improve performance.
5. **Business Integration**  
   - Deploy insights into CRM systems for actionable retention strategies.

---

## Limitations
- Results depend on historical data; may not reflect future behavior.
- Limited predictive power of demographic features.
- Data quality and completeness impact model accuracy.
- Model generalizability may vary for other banks or regions.

---

## Impact and Future Plans
- The model supports **proactive customer retention** strategies.
- Integration into a **Streamlit app** enables business users to predict churn easily.
- Future improvements:
  - Incorporating **economic and transactional trends**.
  - Expanding features with behavioral analytics.
  - Testing other **machine learning algorithms** for enhanced accuracy.

---

## References
- Kaggle customer churn datasets
- XGBoost and Scikit-learn documentation
- Tutorials and articles on supervised machine learning and churn analysis

---

## Contact Information
- **Name:** Satelite Alison Ndayikunda  
- **Email:** [satalisonn@gmail.com](mailto:satalisonn@gmail.com)
