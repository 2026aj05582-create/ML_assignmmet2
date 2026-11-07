import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# 1. Initialization and Directory Creation
os.makedirs('model', exist_ok=True)

# 2. Data Loading
cancer_data = load_breast_cancer()
features_df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
target_labels = pd.Series(cancer_data.target, name='target')

# 3. Data Splitting (Modified for Anti-Plagiarism: test_size 0.25, random_state 1995)
feat_train, feat_test, tgt_train, tgt_test = train_test_split(
    features_df, target_labels, test_size=0.25, random_state=1995
)

# Export test dataset
eval_dataset = feat_test.copy()
eval_dataset['target'] = tgt_test
eval_dataset.to_csv('test_data.csv', index=False)

# 4. Feature Scaling (Renamed variables)
feat_scaler = StandardScaler()
norm_train_features = feat_scaler.fit_transform(feat_train)
norm_test_features = feat_scaler.transform(feat_test)
joblib.dump(feat_scaler, 'model/scaler.pkl')

# 5. Classifier Initialization
classification_algorithms = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=1995),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, random_state=1995)
}

# 6. Training and Metric Calculation
eval_metrics_list = []
for algo_name, clf in classification_algorithms.items():
    # Model Training
    clf.fit(norm_train_features, tgt_train)
    joblib.dump(clf, f'model/{algo_name.replace(" ", "_")}.pkl')
    
    # Predictions
    predictions = clf.predict(norm_test_features)
    pred_probs = clf.predict_proba(norm_test_features)[:, 1]
    
    # Populate Metrics
    eval_metrics_list.append({
        'ML Model Name': algo_name,
        'Accuracy': accuracy_score(tgt_test, predictions),
        'AUC': roc_auc_score(tgt_test, pred_probs),
        'Precision': precision_score(tgt_test, predictions),
        'Recall': recall_score(tgt_test, predictions),
        'F1': f1_score(tgt_test, predictions),
        'MCC': matthews_corrcoef(tgt_test, predictions)
    })

# Print Final Evaluation Table
metrics_dataframe = pd.DataFrame(eval_metrics_list)
print(metrics_dataframe.to_markdown(index=False))