# Student Performance Prediction

A beginner Machine Learning project to predict student performance using Python and Scikit-learn.

## Project Overview

This project uses student data such as:

- Study hours
- Attendance
- Previous marks

A Logistic Regression model is used to predict whether a student is likely to pass or fail.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Jupyter Notebook / VS Code
- GitHub

## Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| study_hours | Number of hours studied |
| attendance | Student attendance percentage |
| previous_marks | Previous academic marks |
| result | 0 = Fail, 1 = Pass |

## Machine Learning Process

1. Load the dataset
2. Prepare the data
3. Split data into training and testing sets
4. Train a Logistic Regression model
5. Predict student performance
6. Evaluate the model
7. Predict the result for a new student

## Example Prediction

The project predicts the performance of a student based on:

- Study Hours: 6
- Attendance: 85%
- Previous Marks: 70

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
