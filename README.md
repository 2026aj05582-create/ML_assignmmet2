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
| Logistic Regression    | 0.973684 | 0.997380 | 0.972222  | 0.985915 | 0.979021 | 0.943898 |
| Decision Tree          | 0.947368 | 0.943990 | 0.957746  | 0.957746 | 0.957746 | 0.887979 |
| KNN                    | 0.947368 | 0.981985 | 0.957746  | 0.957746 | 0.957746 | 0.887979 |
| Naive Bayes            | 0.964912 | 0.997380 | 0.958904  | 0.985915 | 0.972222 | 0.925285 |
| Support Vector Machine | 0.982456 | 0.997380 | 0.972603  | 1 | 0.986111 | 0.962998 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Excellent all-around performance with a near-perfect AUC (~0.997) and balanced Precision/Recall. It proves to be a highly reliable linear model for this dataset. |
| **Decision Tree** | Yields the lowest AUC (~0.944) and ties for the lowest Accuracy (94.7%), suggesting it may be slightly overfitting the training data compared to the other algorithms. |
| **KNN** | Matches the Decision Tree in Accuracy and F1 score but achieves a significantly better AUC (~0.982), indicating much better probability estimates across different classification thresholds. |
| **Naive Bayes** | Performs strongly with a near-perfect AUC (~0.997) and high Recall (~0.986), proving that its assumption of feature independence works surprisingly well here. |
| **Support Vector Machine** | The top performer across the board. It achieves perfect Recall (1.0), meaning it caught every single positive instance in the test set, alongside the highest MCC (~0.963). |
| **Overall, Winner for your dataset?** | **Support Vector Machine**. With the highest Accuracy (98.2%) and perfect Recall, it is the most robust model for this data, offering the best overall balance of precision and sensitivity. |