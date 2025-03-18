import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Decision Tree Classifier Visualization", layout="wide")

# Title and Description
st.title(" Decision Tree Classifier Interactive Visualization")
st.write("Adjust parameters and observe the decision boundary and tree structure in real-time.")
st.markdown('''The dataset used in this application is a synthetic classification dataset generated using make_classification() from Scikit-learn. It consists of:

✅ 200 samples (data points)\n
✅ 2 features (for easy visualization)\n
✅ 2 classes (binary classification: Class 0 & Class 1)\n
✅ No redundant features (ensuring clear decision boundaries)

This dataset helps in demonstrating how the Decision Tree Classifier splits the data based on feature values and forms decision boundaries.

Since the dataset is synthetic, it allows easy control over scaling, complexity, and visualization of decision boundaries!''')

# Sidebar Parameters
st.sidebar.header("🔧Tune the Parameters")

# Option to choose between scaled and unscaled data
scale_option = st.sidebar.selectbox("Apply Standard Scaling?", ["Yes", "No"])

# Criterion 
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

# Max Depth
max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=20, value=3, step=1)



# Min Samples Split
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)

# Min Samples Leaf
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)

# Generate Synthetic Dataset
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# Apply Standard Scaler if chosen
if scale_option == "Yes":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scale_status = "Standardized"
else:
    scale_status = "Raw Data"

# Train Decision Tree Model
model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, 
                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
model.fit(X, y)

# Function to Plot Decision Boundary
def plot_decision_boundary():
    # Create Mesh Grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot Decision Boundary
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, edgecolor='black', s=80, palette='coolwarm', ax=ax)

    ax.set_title(f"Decision Tree Boundary (Depth={max_depth}, {scale_status})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    return fig

# Function to Plot the Decision Tree Structure
def plot_decision_tree():
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, filled=True, feature_names=["Feature 1", "Feature 2"], class_names=["Class 0", "Class 1"], ax=ax)
    ax.set_title(f"Decision Tree Structure (Depth={max_depth})")
    return fig

# Display Decision Boundary
st.subheader("🛠 Decision Boundary")
st.pyplot(plot_decision_boundary())

# Display Decision Tree Structure
st.subheader("🌳 Decision Tree Structure")
st.pyplot(plot_decision_tree())
