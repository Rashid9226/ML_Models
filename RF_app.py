import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Page config
st.set_page_config(page_title="DT vs RF Classifier", layout="wide")
st.title("🌿 Decision Tree vs Random Forest - Binary Classification")

# Generate dataset
X, y = make_classification(
    n_samples=300, 
    n_features=2, 
    n_informative=2, 
    n_redundant=0, 
    n_clusters_per_class=1, 
    random_state=42
)

# Sidebar model selection
st.sidebar.header("Choose Classifier")
model_type = st.sidebar.selectbox("Model", ["Decision Tree", "Random Forest"])
test_size = st.sidebar.slider("Test Size", 0.2, 0.5, 0.3)

# Model parameters
if model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    show_oob = False
else:
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        oob_score=True,
        bootstrap=True,
        random_state=42
    )
    show_oob = True

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
st.subheader(f"📊 Accuracy: `{accuracy_score(y_test, y_pred):.2f}`")

# OOB score for Random Forest
if show_oob:
    st.subheader(f"🧪 OOB Score: `{model.oob_score_:.2f}`")

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
fig1, ax1 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax1)
st.pyplot(fig1)

# Decision Boundary
st.subheader("🌀 Decision Boundary")
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig2, ax2 = plt.subplots()
ax2.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
ax2.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=40, cmap=plt.cm.coolwarm)
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.set_title(f"Decision Boundary - {model_type}")
st.pyplot(fig2)
