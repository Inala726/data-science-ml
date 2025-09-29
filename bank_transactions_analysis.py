import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the dataset
data = pd.read_csv('bank_transactions_data_2.csv')


# print(data.head())
# print(data.shape)

# print(data.info())
# print(data.describe())




# Missing values
# missing = data.isnull().sum()
# missing_percent = (missing / len(data)) * 100
# missing_data = pd.DataFrame({
#     'Missing Values': missing,
#     'Percentage': missing_percent
# })
# print(data.duplicated().sum())
# print(missing_data)

# # duplicate rows
# print(data.duplicated().sum())

# # Column type separation
# id_cols = ['TransactionID', 'AccountID']
# date_cols = ['TransactionDate', 'PreviousTransactionDate']

# # Convert to datetime
# for col in date_cols:
#     if col in data.columns:
#         data[col] = pd.to_datetime(data[col], errors='coerce')

# # Identify numeric vs categorical
# numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
# categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# print("Numeric columns:", numeric_cols)
# print("Categorical columns:", categorical_cols)

# EXPLANATORY DATA ANALYSIS (EDA)
# Distribution of transaction amounts
# plt.figure(figsize=(8,5))
# sns.histplot(data['TransactionAmount'], bins=50, kde=True)
# plt.title("Distribution of Transaction Amounts")
# plt.xlabel("Transaction Amount")
# plt.ylabel("Frequency")
# plt.show()

# Boxplot to highlight outliers
# plt.figure(figsize=(6,5))
# sns.boxplot(y=data['TransactionAmount'])
# plt.title("Boxplot of Transaction Amounts")
# plt.ylabel("Transaction Amount")
# plt.show()

# Distribution of customer ages
# plt.figure(figsize=(8,5))
# sns.histplot(data['CustomerAge'], bins=20, kde=True)
# plt.title("Distribution of Customer Age")
# plt.xlabel("Customer Age")
# plt.ylabel("Frequency")
# plt.show()

# # Boxplot
# plt.figure(figsize=(6,5))
# sns.boxplot(y=data['CustomerAge'])
# plt.title("Boxplot of Customer Age")
# plt.ylabel("Customer Age")
# plt.show()

# Distribution of transaction duration
# plt.figure(figsize=(8,5))
# sns.histplot(data['TransactionDuration'], bins=20, kde=True)
# plt.title("Distribution of Transaction Duration")
# plt.xlabel("Transaction Duration (seconds)")
# plt.ylabel("Frequency")
# plt.show()

# # Boxplot
# plt.figure(figsize=(6,5))
# sns.boxplot(y=data['TransactionDuration'])
# plt.title("Boxplot of Transaction Duration")
# plt.ylabel("Transaction Duration (seconds)")
# plt.show()

# Distribution of login attempts
# plt.figure(figsize=(8,5))
# sns.countplot(x='LoginAttempts', data=data)
# plt.title("Distribution of Login Attempts")
# plt.xlabel("Number of Login Attempts")
# plt.ylabel("Count")
# plt.show()

# # Boxplot
# plt.figure(figsize=(6,5))
# sns.boxplot(y=data['LoginAttempts'])
# plt.title("Boxplot of Login Attempts")
# plt.ylabel("Login Attempts")
# plt.show()

# Distribution of account balances
# plt.figure(figsize=(8,5))
# sns.histplot(data['AccountBalance'], bins=30, kde=True)
# plt.title("Distribution of Account Balances")
# plt.xlabel("Account Balance")
# plt.ylabel("Frequency")
# plt.show()

# # Boxplot to check for outliers
# plt.figure(figsize=(6,5))
# sns.boxplot(y=data['AccountBalance'])
# plt.title("Boxplot of Account Balances")
# plt.ylabel("Account Balance")
# plt.show()

# cat_cols = [
#     'TransactionType', 
#     'Location', 
#     'Channel', 
#     'CustomerOccupation'
# ]

# for col in cat_cols:
#     plt.figure(figsize=(12,5))
#     order = data[col].value_counts().index
#     sns.countplot(x=col, data=data, order=order, palette="viridis")
#     plt.title(f"{col} Distribution")
#     plt.xticks(rotation=45)
#     plt.show()

# plt.figure(figsize=(8,6))
# sns.boxplot(x='TransactionType', y='TransactionAmount', data=data)
# plt.title("Transaction Amount by Transaction Type")
# plt.xlabel("Transaction Type")
# plt.ylabel("Transaction Amount")
# plt.show()

# plt.figure(figsize=(10,6))
# sns.boxplot(x='CustomerOccupation', y='AccountBalance', data=data)
# plt.title("Account Balance by Customer Occupation")
# plt.xlabel("Customer Occupation")
# plt.ylabel("Account Balance")
# plt.xticks(rotation=45)
# plt.show()

# plt.figure(figsize=(8,6))
# sns.boxplot(x='Channel', y='LoginAttempts', data=data)
# plt.title("Login Attempts by Transaction Channel")
# plt.xlabel("Transaction Channel")
# plt.ylabel("Login Attempts")
# plt.show()

# correlation heatmap between numerical columsns
# Select numeric features only
# numeric_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts', 'AccountBalance']

# # Correlation matrix
# corr = data[numeric_cols].corr()

# # Heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap of Numeric Features")
# plt.show()

# Feature Engineering
data_model = data.copy()
drop_cols = [
    "TransactionID", "AccountID", "TransactionDate",
    "PreviousTransactionDate", "IP Address", "Amount_Z_Score"
]
data_model.drop(columns=[c for c in drop_cols if c in data_model.columns], inplace=True)
data_model

avg_tx_amount_by_type = data_model.groupby("TransactionType")["TransactionAmount"].transform("mean")
data_model["Amount_to_AvgByType_Ratio"] = data_model["TransactionAmount"] / avg_tx_amount_by_type

# Device transaction count
device_tx_count = data_model.groupby("DeviceID").size().reset_index(name="DeviceTxCount")
data_model = data_model.merge(device_tx_count, on="DeviceID", how="left")

categorical_cols = data_model.select_dtypes(include=["object"]).columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    data_model[col] = le.fit_transform(data_model[col])

categorical_cols = data_model.select_dtypes(include=["object"]).columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    data_model[col] = le.fit_transform(data_model[col])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(data_model)
df_scaled_df = pd.DataFrame(df_scaled, columns=data_model.columns)

print(f"Final dataset shape: {data_model.shape}")