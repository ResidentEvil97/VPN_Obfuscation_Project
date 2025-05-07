# ml-based mock dpi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the data
df = pd.read_csv('dpi/data/sample_combined_balanced.csv')

# Encode label: VPN=1, Non-VPN=0
df['Label'] = LabelEncoder().fit_transform(df['Label'])

# Select features and label
X = df[['duration', 'flowPktsPerSecond', 'flowBytesPerSecond']]
y = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate (optional)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", clf.score(X_test, y_test))
print(f"Model accuracy: {accuracy:.2%}")

# Save the model
joblib.dump(clf, "dpi/models/random_forest_dpi.pkl")
print("Model saved to dpi/models/random_forest_dpi.pkl")

