# Machine Learning Assignment 2

## a. Problem statement
The objective of this project is to build, evaluate, and deploy five different machine learning classification models to accurately predict whether a breast mass is malignant or benign based on digitized image features of a fine needle aspirate (FNA) of a breast mass. The project demonstrates an end-to-end ML deployment workflow, including modeling, evaluation, UI design, and deployment via Streamlit Community Cloud.

## b. Dataset description
*   **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
*   **Source:** Scikit-learn datasets (originally from the UCI Machine Learning Repository)
*   **Features:** 30 numeric, predictive attributes (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity).
*   **Instances:** 569
*   **Target:** Binary classification (0 = Malignant, 1 = Benign)

## c. Github Repository Link
https://github.com/2026aj05582-create/ML_assignmmet2.git

## d. Models used
### Comparison Table

| ML Model Name          | Accuracy | AUC      | Precision | Recall   | F1       | MCC      |
|:-----------------------|---------:|---------:|----------:|---------:|---------:|---------:|
| Logistic Regression    | 0.986014 | 0.997727 | 0.977778  | 1.000000 | 0.988764 | 0.970681 |
| Decision Tree          | 0.916084 | 0.928409 | 0.987179  | 0.875000 | 0.927711 | 0.837158 |
| KNN                    | 0.979021 | 0.996694 | 0.967033  | 1.000000 | 0.983240 | 0.956183 |
| Naive Bayes            | 0.958042 | 0.996074 | 0.965909  | 0.965909 | 0.965909 | 0.911364 |
| Support Vector Machine | 0.993007 | 0.999587 | 0.988764  | 1.000000 | 0.994350 | 0.985285 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Exceptional all-around performance, handling the linearly separable features well. Perfect recall (1.0) and high AUC (~0.997) make it a strong candidate for medical data where missing a positive case is costly. |
| **Decision Tree** | Yields the lowest AUC (~0.928), Accuracy (~91.6%), and Recall (~87.5%). This suggests a strong tendency to overfit the training splits compared to distance-based or probability-based algorithms. |
| **KNN** | Very solid baseline that benefits from the standard scaling, correctly identifying nearest neighbors in the feature space and achieving perfect recall (1.0) with an accuracy of ~97.9%. |
| **Naive Bayes** | Performs remarkably well, boasting a high AUC (~0.996) and perfectly balanced precision and recall (~0.965) despite its assumption that all features are independent. |
| **Support Vector Machine** | Dominates across most metrics. The kernel accurately separates the data, capturing complex decision boundaries to achieve near-perfect AUC (~0.999) and perfect recall (1.0). |
| **Overall, Winner for your dataset?** | **Support Vector Machine**. It yields the highest overall Accuracy (~99.3%), F1 Score (~0.994), and MCC (~0.985) in this split, balancing precision and recall perfectly to minimize critical false negatives. |