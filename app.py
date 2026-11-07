import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Classification Evaluator")
st.write("Upload your test data to evaluate the pre-trained classification models.")

# Load scaler
@st.cache_resource
def load_scaler():
    return joblib.load('model/scaler.pkl')

scaler = load_scaler()

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload test_data.csv", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Assuming last column is target as saved by train_models.py
    X_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1]
    
    X_test_scaled = scaler.transform(X_test)
    
    # b. Model selection dropdown
    model_choice = st.selectbox(
        "Select a Machine Learning Model",
        ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
    )
    
    if st.button("Evaluate Model"):
        # Load selected model
        model_filename = f'model/{model_choice.replace(" ", "_")}.pkl'
        model = joblib.load(model_filename)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # c. Display of evaluation metrics
        st.write(f"### Evaluation Metrics for {model_choice}")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        # d. Confusion matrix or classification report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)