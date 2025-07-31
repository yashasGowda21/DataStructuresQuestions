# ✅ Full Example to Practice All 20 Pandas Interview Scenarios

import pandas as pd
import numpy as np

# ---------------------------
# 🔢 1. Create Sample DataFrame
# ---------------------------
data = {
    '  Name  ': ['  Alice  ', 'Bob ', ' Charlie', '"David"', "'Eve'", 'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy', 'Bob '],
    'Age': [25, 30, np.nan, 45, 28, 30, 32, 29, 41, 33, 30],
    'Department': ['HR', 'IT', 'HR', 'Finance', 'IT', 'IT', 'Finance', 'HR', np.nan, 'Finance', 'IT'],
    'Salary': [50000, 60000, 52000, 80000, 62000, 60000, 75000, np.nan, 71000, 69000, 60000],
    'JoinDate': ['2015-03-01', '2016-07-15', '2017-01-10', '2014-11-23', '2018-09-05', '2016-07-15', '2014-11-23', '2017-01-10', '2019-06-30', '2013-05-20', '2016-07-15'],
    'HasLeft': [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# ---------------------------------
# 🧹 2. Strip column names (rename)
# ---------------------------------
df.columns = df.columns.str.strip()

# ----------------------------------------------------
# 🧽 3. Clean 'Name' column (strip whitespace, quotes)
# ----------------------------------------------------
df['Name'] = df['Name'].str.strip(" '\"")

# ----------------------------------
# 🤕 4. Detect & fill missing values
# ----------------------------------
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Department'] = df['Department'].fillna('Unknown')

# --------------------------------
# 🧯 5. Remove duplicate rows
# --------------------------------
df = df.drop_duplicates()

# --------------------------------
# 🔧 6. Convert column data types
# --------------------------------
df['JoinDate'] = pd.to_datetime(df['JoinDate'])

# -------------------------------
# 🏷️ 7. Encode categorical column
# -------------------------------
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# -----------------------------------------
# 🔍 8. Filter rows based on conditions
# -----------------------------------------
adults = df[df['Age'] > 30]

# ----------------------------------
# 🔁 9. Apply function to a column
# ----------------------------------
df['NameLength'] = df['Name'].apply(len)

# -------------------------------
# 📊 10. Group by and aggregate
# -------------------------------
salary_by_status = df.groupby('HasLeft')['Salary'].mean()

# -----------------------------------
# 🚨 11. Outlier handling using IQR
# -----------------------------------
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
outlier_free_df = df[(df['Salary'] >= Q1 - 1.5 * IQR) & (df['Salary'] <= Q3 + 1.5 * IQR)]

# ---------------------------------
# 🔢 12. Use loc and iloc examples
# ---------------------------------
row_1 = df.iloc[0]         # First row
row_named = df.loc[0]     # Row with label/index 0

# -----------------------------------
# 🧬 13. Merge/join with another DataFrame
# -----------------------------------
dep_info = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Location': ['NY', 'LA', 'TX', 'NY', 'SF', 'LA']
})

merged_df = pd.merge(df, dep_info, on='Name', how='left')

# -----------------------------------
# 🧱 14. Concatenate another DataFrame
# -----------------------------------
df2 = df.copy()
concatenated_df = pd.concat([df, df2], axis=0)

# --------------------------------------
# 📏 15. Normalize numerical columns
# --------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# -----------------------------
# 📌 16. Sort the DataFrame
# -----------------------------
df_sorted = df.sort_values(by='Salary', ascending=False)

# ------------------------------------
# 📉 17. Outliers using Z-score method
# ------------------------------------
from scipy import stats
z_scores = np.abs(stats.zscore(df[['Salary']]))
z_filtered = df[(z_scores < 3).all(axis=1)]

# ----------------------------------
# 🔄 18. Reset index
# ----------------------------------
df_reset = df.reset_index(drop=True)

# ----------------------------------
# 🔄 19. Pivot and unpivot
# ----------------------------------
pivot = df.pivot_table(index='HasLeft', values='Salary', aggfunc='mean')
melted = pd.melt(df, id_vars=['Name'], value_vars=['Age', 'Salary'])

# ----------------------------------
# 🔍 20. Use apply/map/applymap
# ----------------------------------
df['UpperName'] = df['Name'].map(str.upper)

# Applymap on DataFrame (only works for all-string/numeric DataFrame)
name_df = df[['Name']].applymap(str.upper)

# ✅ Final cleaned and feature-rich DataFrame is in df
