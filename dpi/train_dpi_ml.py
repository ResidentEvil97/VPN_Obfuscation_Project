# ml-based mock dpi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

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
print("Test accuracy:", clf.score(X_test, y_test))

# Save the model
joblib.dump(clf, "dpi/data/dpi_model.joblib")
print("Model saved to dpi/data/dpi_model.joblib")

