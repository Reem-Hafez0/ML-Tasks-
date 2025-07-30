
# Machine Learning Projects

This repository contains multiple machine learning tasks. Each task includes preprocessing steps, model training, evaluation, and optional enhancements to improve performance.


## Task 1: Traffic Sign Recognition

**Description**:  
Build a deep learning model to classify traffic signs based on images.

**Dataset**:  
GTSRB Dataset - Kaggle

**Objectives**:
- Preprocess images (resizing, normalization)
- Train a CNN model to recognize traffic sign classes
- Evaluate using accuracy and confusion matrix

**Tools & Libraries**:
- Python
- TensorFlow
- Keras
- OpenCV

**Covered Topics**:
- Computer Vision (CNN)
- Multi-class classification

---

## Task 2: Forest Cover Type Classification

**Description**:  
Predict the type of forest cover based on cartographic and environmental features.

**Dataset**:  
Covertype Dataset - UCI

**Objectives**:
- Clean and preprocess data (including categorical features)
- Train and evaluate multi-class classification models
- Visualize confusion matrix and feature importance

**Tools & Libraries**:
- Python
- Pandas
- Scikit-learn
- XGBoost

**Covered Topics**:
- Multi-class classification
- Tree-based modeling
- Compare different models (e.g., Random Forest vs. XGBoost)
- Perform hyperparameter tuning

---

## Task 3: Loan Approval Prediction

**Description**:  
Build a classification model to predict whether a loan application will be approved.

**Dataset**:  
Loan Approval Prediction Dataset - Kaggle

**Objectives**:
- Handle missing values and encode categorical features
- Train a classification model on imbalanced data
- Evaluate using precision, recall, and F1-score

**Tools & Libraries**:
- Python
- Pandas
- Scikit-learn

**Covered Topics**:
- Binary classification
- Handling imbalanced data
- Use SMOTE to handle class imbalance
- Compare Logistic Regression vs. Decision Tree


---

## Project Structure

```bash
ml-projects/
 ┣ Forest_Cover_Type/
 ┃ ┣ Forest_Cover_Type_Task.ipynb
 ┃ ┣ covtype.csv
 ┃ ┗ ...
 ┣ Loan_Approval_Prediction/
 ┃ ┣ Loan_Approval_Prediction_Task.ipynb
 ┃ ┣ loan_prediction.csv
 ┃ ┗ ...
 ┣ Traffic_Sign_Recognition/
 ┃ ┣ Traffic_Sign_Recognition_Task.ipynb
 ┃ ┣ GTSRB
 ┃ ┗ ...
 ┗ README.md
```
