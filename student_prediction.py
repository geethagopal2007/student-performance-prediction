import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV
df = pd.read_csv("student_performance.csv")

print("Student Dataset:")
print(df)

# Features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict a new student
new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [85],
    "previous_marks": [70]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nPrediction: Student is likely to PASS")
else:
    print("\nPrediction: Student is likely to FAIL")
