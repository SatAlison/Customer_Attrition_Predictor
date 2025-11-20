
## **Project Description**
This dataset is focused on helping a bank identify which customers are likely to churn—i.e., stop using their credit card services. With a churn rate of approximately 16%, it presents a class imbalance problem that makes modeling challenging. The dataset contains information on customer demographics, credit card usage patterns, relationship metrics, and derived features. The goal is to use this data to build a predictive model that flags potential churners so the bank can take proactive retention actions.
Column Descriptions
●	**CLIENTNUM** – Unique identifier assigned to each customer .

●	**Attrition_Flag** – Indicates whether the customer is an Existing Customer or an Attrited Customer (churned).

●	**Customer_Age** – Age of the customer.

●	**Gender** – Gender of the customer (M or F).

●	**Dependent_count** – Number of dependents the customer has.

●	**Education_Level** – Education qualification (e.g., High School, College, Doctorate).

●	**Marital_Status** – Marital status of the customer (e.g., Married, Single).

●	**Income_Category** – Annual income range of the customer.

●	**Card_Category** – Type of credit card held (e.g., Blue, Silver, Platinum).

●	**Months_on_book** – Number of months the account has been open.

●	**Total_Relationship_Count** – Total number of products/services the customer uses with the bank.

●	**Months_Inactive_12_mon** – Number of months the customer was inactive in the past 12 months.

●	**Contacts_Count_12_mon** – Number of contacts (e.g., calls, emails) made with the customer in the last 12 months.

●	**Credit_Limit** – Credit limit assigned to the customer.

●	**Total_Revolving_Bal** – Total revolving balance on the credit card.

●	**Avg_Open_To_Buy** – Average open-to-buy amount on the credit card.

●	**Total_Amt_Chng_Q4_Q1** – Ratio of transaction amount change from Q1 to Q4.

●	**Total_Trans_Amt** – Total transaction amount over the last 12 months.

●	**Total_Trans_Ct** – Total transaction count over the last 12 months.

●	**Total_Ct_Chng_Q4_Q1** – Ratio of transaction count change from Q1 to Q4.

●	**Avg_Utilization_Ratio** – Average credit card utilization ratio.

●	**Naive_Bayes_Classifier_**..._1 – Predicted churn probability (class 1) from a pre-built Naive Bayes model.

●	**Naive_Bayes_Classifier_**..._2 – Predicted churn probability (class 2) from a pre-built Naive Bayes model.

#### **Limitations**


●	Class imbalance: Only ~16% of customers have churned, which can bias models toward predicting "non-churn".

●	Missing domain context: Variables like Total_Amt_Chng_Q4_Q1 or Avg_Open_To_Buy are helpful but may lack interpretability for stakeholders without business context.

●	Naive Bayes columns: These may need to be excluded from modeling if they are derived from the same data and model you are trying to improve upon.


## **Problem Statement**
Can we predict whether a customer will churn based on their profile, engagement metrics, and spending behavior?
Approaches

●	Exploratory Data Analysis (EDA) to understand patterns across age, income, credit usage, etc.

●	Feature Engineering to handle class imbalance (SMOTE, under/oversampling).

●	Classification Models such as Logistic Regression, Random Forest, XGBoost, and LightGBM.

●	Model Evaluation using metrics like ROC-AUC, F1-score, and Precision-Recall curves due to imbalance.

Use Cases
●	Customer Retention Strategy: Identify high-risk customers and target them with loyalty programs.

●	Churn Score Deployment: Implement predictive churn scoring in customer dashboards.

●	Marketing Optimization: Focus outreach on customers likely to churn to reduce marketing spend.
