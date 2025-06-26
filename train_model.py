import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load dataset
df = pd.read_csv("creditcard.csv")

# Step 2: Preprocess
df['NormalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Step 3: Split data
X = df.drop('Class', axis=1)
y = df['Class']

# Step 4: Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Step 5: Train model
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 7: Save model
pickle.dump(model, open("fraud_model.pkl", "wb"))
