import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1️⃣ GENERATE DUMMY DATA
np.random.seed(42)

num_customers = 1000
customer_ids = np.arange(1, num_customers + 1)
loan_types = ['Home Loan', 'Car Loan', 'Personal Loan', 'Education Loan', 'Business Loan']
preferred_loans = np.random.choice(loan_types, num_customers)
loan_amounts = np.random.randint(5000, 100000, num_customers)
last_loan_date = [datetime(2020, 1, 1) + timedelta(days=random.randint(30, 1000)) for _ in range(num_customers)]

data = pd.DataFrame({
    'Customer_ID': customer_ids,
    'Loan_Type': preferred_loans,
    'Loan_Amount': loan_amounts,
    'Last_Loan_Date': last_loan_date,
    'Income': np.random.randint(30000, 200000, num_customers),
    'Credit_Score': np.random.randint(300, 900, num_customers),
    'Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], num_customers),
    'Age': np.random.randint(22, 65, num_customers)
})

# Convert Employment_Status to numeric
data['Employment_Status'] = data['Employment_Status'].map({'Employed': 2, 'Self-Employed': 1, 'Unemployed': 0})

# 2️⃣ CUSTOMER SEGMENTATION USING K-MEANS
X = data[['Income', 'Credit_Score', 'Employment_Status', 'Age']]
kmeans = KMeans(n_clusters=5, random_state=42)
data['Segment'] = kmeans.fit_predict(X)

# 3️⃣ PREDICTING NEXT 3 MOST PREFERABLE LOAN TYPES
X_loan = data[['Income', 'Credit_Score', 'Employment_Status', 'Age']]
y_loan = data['Loan_Type']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_loan, y_loan)
data['Predicted_Loan_1'] = rf_model.predict(X_loan)

loan_probabilities = pd.DataFrame(rf_model.predict_proba(X_loan), columns=rf_model.classes_)
data['Predicted_Loan_2'] = loan_probabilities.apply(lambda x: loan_types[np.argsort(x)[-2]], axis=1)
data['Predicted_Loan_3'] = loan_probabilities.apply(lambda x: loan_types[np.argsort(x)[-3]], axis=1)

# 4️⃣ PREDICTING PREFERABLE LOAN AMOUNT
X_amount = data[['Income', 'Credit_Score', 'Employment_Status', 'Age']]
y_amount = data['Loan_Amount']

lr_model = LinearRegression()
lr_model.fit(X_amount, y_amount)
data['Predicted_Loan_Amount'] = lr_model.predict(X_amount)

# 5️⃣ BEST TIME TO PITCH NEXT LOAN OFFER
data['Days_Since_Last_Loan'] = (datetime.today() - data['Last_Loan_Date']).dt.days
time_series_data = data.groupby('Segment')['Days_Since_Last_Loan'].mean()

best_time_model = ExponentialSmoothing(time_series_data, trend='add', seasonal=None).fit()
next_offer_days = best_time_model.forecast(1).iloc[0]

data['Next_Offer_Days'] = next_offer_days

# 6️⃣ DISPLAY RESULTS
print(data[['Customer_ID', 'Segment', 'Predicted_Loan_1', 'Predicted_Loan_2', 'Predicted_Loan_3', 
            'Predicted_Loan_Amount', 'Next_Offer_Days']].head(10))
