# Student Performance Prediction using Machine Learning

A machine learning project that predicts student academic performance based on behavioral and academic features using Linear Regression and Neural Networks.

---

## Overview

This project builds and compares two ML models to predict a student's final exam score based on:
- Study hours per day
- Attendance percentage
- Previous grades
- Assignments completed
- Sleep hours

## Results

| Model | R2 Score | Accuracy (within 10 marks) |
|-------|----------|---------------------------|
| Linear Regression | ~0.87 | ~87% |
| Neural Network (MLP) | ~0.91 | ~91% |

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** Scikit-learn, Pandas, NumPy
- **Algorithms:** Linear Regression, MLP Neural Network
- **Preprocessing:** StandardScaler, Train-Test Split

## How to Run

```bash
# Clone the repo
git clone https://github.com/Jagrutiz/student-performance-prediction

# Install dependencies
pip install pandas numpy scikit-learn

# Run the model
python student_performance.py
```

## Project Structure

```
student-performance-prediction/
│
├── student_performance.py   # Main ML script
├── README.md                # Project documentation
```

## Key Learnings

- Data preprocessing (normalization, feature selection)
- Comparing ML models using R2 and RMSE metrics
- Hyperparameter tuning for Neural Networks
- Making predictions on unseen data

---

**Author:** Jagruti Pankaj Zunjarrao  
**Email:** jagrutizunjarrao31@gmail.com
