import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch the UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Combine features and targets
df = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)

# Rename target if needed
if 'target' not in df.columns:
    if 'num' in df.columns:
        df = df.rename(columns={'num': 'target'})
    else:
        raise ValueError("No 'target' or 'num' column found in dataframe columns!")

# Convert target to binary: 1 = disease present (values 1-4), 0 = no disease (value 0)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Drop rows with missing values for simplicity (or use imputation)
df = df.dropna()

# Save cleaned CSV
df.to_csv('data/heart_clean.csv', index=False)
print("Cleaned data saved to data/heart_clean.csv")