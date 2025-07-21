import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="ML Model Explorer", layout="wide")

st.title("Machine Learning Model Explorer")
st.write("This app demonstrates how to build and evaluate ML models with Streamlit.")

# Sidebar for dataset selection and model parameters
st.sidebar.header("Settings")

# Dataset selection
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Breast Cancer"))


# Load selected dataset
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Wine":
        data = load_wine()
    else:
        data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.target_names


X, y, class_names = get_dataset(dataset_name)

# Display dataset info
st.header("Dataset Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Dataset:**", dataset_name)
    st.write("**Samples:**", X.shape[0])
with col2:
    st.write("**Features:**", X.shape[1])
    st.write("**Classes:**", len(class_names))
with col3:
    st.write("**Class Names:**", class_names)
    st.write("**Class Distribution:**", np.bincount(y))

# Display sample data
with st.expander("Show Sample Data"):
    st.dataframe(pd.concat([X, y], axis=1).head(10))

# Feature selection
st.subheader("Feature Selection")
if X.shape[1] > 10:
    top_n = st.slider("Select number of features to use", 2, X.shape[1], min(10, X.shape[1]))
    feature_imp = RandomForestClassifier().fit(X, y).feature_importances_
    indices = np.argsort(feature_imp)[::-1]
    selected_features = X.columns[indices[:top_n]].tolist()
else:
    selected_features = st.multiselect("Select features to include", X.columns.tolist(), default=X.columns[:3].tolist())

if len(selected_features) < 2:
    st.error("Please select at least 2 features.")
    st.stop()

X_selected = X[selected_features]

# Model selection and parameters
st.sidebar.header("Model Configuration")
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Logistic Regression", "Random Forest", "SVM", "Gradient Boosting")
)


# Set up model parameters based on selection
def get_classifier_params(clf_name):
    params = {}
    if clf_name == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        max_iter = st.sidebar.slider("Maximum Iterations", 100, 1000, 100)
        params["C"] = C
        params["max_iter"] = max_iter
    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of trees", 10, 300, 100)
        max_depth = st.sidebar.slider("Maximum depth", 2, 20, 5)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif clf_name == "SVM":
        C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))
        params["C"] = C
        params["kernel"] = kernel
    else:  # Gradient Boosting
        n_estimators = st.sidebar.slider("Number of trees", 10, 300, 100)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate

    return params


params = get_classifier_params(classifier_name)

# Train-test split
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
random_state = st.sidebar.slider("Random State", 0, 100, 42)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state)

# Scaling
do_scaling = st.sidebar.checkbox("Scale Features", True)
if do_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.info("Features have been standardized.")

# Model training
st.header("Model Training and Evaluation")


def get_classifier(clf_name, params):
    if clf_name == "Logistic Regression":
        return LogisticRegression(**params)
    elif clf_name == "Random Forest":
        return RandomForestClassifier(**params)
    elif clf_name == "SVM":
        return SVC(**params, probability=True)
    else:  # Gradient Boosting
        return GradientBoostingClassifier(**params)


clf = get_classifier(classifier_name, params)

if st.button("Train Model"):
    with st.spinner("Training model..."):
        clf.fit(X_train, y_train)

    st.success(f"Model trained! ({classifier_name})")

    # Model evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        st.pyplot(fig)

    # Feature importance (if available)
    if hasattr(clf, "feature_importances_"):
        st.subheader("Feature Importance")
        importances = clf.feature_importances_
        feature_imp = pd.DataFrame({"Feature": X_selected.columns, "Importance": importances}).sort_values(
            "Importance", ascending=False
        )

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=feature_imp, ax=ax)
        st.pyplot(fig)

    # Interactive prediction
    st.subheader("Interactive Prediction")
    st.write("Adjust feature values to get a prediction:")

    input_features = {}
    for feature in X_selected.columns:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        input_features[feature] = st.slider(f"{feature}", min_val, max_val, mean_val)

    input_df = pd.DataFrame([input_features])
    if do_scaling:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df

    prediction = clf.predict(input_scaled)[0]
    pred_proba = clf.predict_proba(input_scaled)[0]

    st.write("Prediction:", class_names[prediction])
    st.write("Prediction Probabilities:")

    proba_df = pd.DataFrame({"Class": class_names, "Probability": pred_proba})

    fig, ax = plt.subplots()
    sns.barplot(x="Class", y="Probability", data=proba_df, ax=ax)
    st.pyplot(fig)
