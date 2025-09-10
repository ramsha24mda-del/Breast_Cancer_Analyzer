import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Load dataset ---
csv_path = "Breast_cancer_data.csv"  # Make sure this CSV is in the project folder
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found! Place your CSV in the project folder.")

df = pd.read_csv(csv_path)

# --- Features & Target ---
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Save Model ---
os.makedirs("model", exist_ok=True)  # create model folder if not exists
model_path = "model/cancer.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model trained & saved as {model_path}")
