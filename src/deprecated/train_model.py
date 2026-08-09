import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("features/dataset.csv")
print("\nDataset Preview:")
print(df.head())

# Inputs
X = df[[
    "mean_hue",
    "mean_blue",
    "mean_green",
    "mean_red",
    "mean_saturation",
    "mean_value",
    "hue_std",
    "saturation_std",
    "value_std",
    "brightness",
    "contrast"
]]

# Labels
y = df["label"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)


print("Dataset size:", len(df))
print(df["label"].value_counts())

print("Accuracy:", accuracy)
print("Actual:")
print(list(y_test))

print("Predicted:")
print(list(predictions))
print("\nFeature Importance")

for name, coef in zip(X.columns, model.coef_[0]):
    print(name, ":", coef)


