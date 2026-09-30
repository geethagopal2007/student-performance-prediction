import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample student dataset
data = {
    "study_hours": [2, 3, 5, 6, 8, 1, 4, 7, 3, 9],
    "attendance": [60, 65, 80, 85, 95, 50, 75, 90, 70, 98],
    "previous_marks": [45, 50, 65, 70, 85, 35, 60, 80, 55, 90],
    "result": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict a new student's result
new_student = [[6, 85, 70]]
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nPrediction: Student is likely to PASS")
else:
    print("\nPrediction: Student is likely to FAIL")
