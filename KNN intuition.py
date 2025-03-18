import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="KNN Classifier with New Data", layout="wide")

# Title and Description
st.title("🔍 K-Nearest Neighbors (KNN) Classifier - Interactive Visualization")
st.write("Adjust the number of neighbors (K), add a new data point, and observe how it gets classified!")
st.markdown('''The dataset used in this visualization is the Moons Dataset, generated using make_moons() from sklearn.datasets. It is commonly used for classification tasks and is ideal for demonstrating machine learning algorithms like K-Nearest Neighbors (KNN).

🟢 Key Characteristics:\n
✅ Two classes (Class 0 & Class 1) forming a crescent moon shape.\n
✅ Non-linearly separable → KNN works well as it adapts to the shape of data.\n
✅ Includes noise → Random variations make classification more realistic.\n
✅ Useful for testing decision boundaries in machine learning models.\n

The data provides a challenging decision boundary for KNN. Changes in K (number of neighbors) directly impact how well the model adapts. Works well with Standard Scaling to observe differences between raw and scaled data.''')

# Sidebar: Adjust K value
st.sidebar.header("🔧 Adjust KNN Parameter")
n_neighbors = st.sidebar.slider("Number of Neighbors (K)", min_value=1, max_value=20, value=5, step=1)

# Option to choose between scaled and unscaled data
scale_option = st.sidebar.radio("Apply Standard Scaling?", ["Yes", "No"])

# Generate Synthetic Dataset (Moons Dataset)
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Apply Standard Scaler if chosen
if scale_option == "Yes":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scale_status = "Standardized"
else:
    scale_status = "Raw Data"

# Train KNN Model
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X, y)

# Sidebar: Add a new data point
st.sidebar.header("📌 Add a New Data Point")
new_x = st.sidebar.slider("New Data X-Coordinate", float(X[:, 0].min()), float(X[:, 0].max()), 0.0, step=0.1)
new_y = st.sidebar.slider("New Data Y-Coordinate", float(X[:, 1].min()), float(X[:, 1].max()), 0.0, step=0.1)

# Classify new data point
new_data = np.array([[new_x, new_y]])
new_class = model.predict(new_data)[0]

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

    # Plot new data point
    ax.scatter(new_x, new_y, color='yellow', edgecolor='black', s=150, marker='*', label="New Data")

    ax.set_title(f"KNN Decision Boundary (K={n_neighbors}, {scale_status})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()

    return fig

# Display Decision Boundary
st.subheader("🛠 Decision Boundary with New Data")
st.pyplot(plot_decision_boundary())

# Show classification result
st.write(f"📌 **New Data Point ({new_x:.2f}, {new_y:.2f}) is classified as:** {'Class 0' if new_class == 0 else 'Class 1'}")
