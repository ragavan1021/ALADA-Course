import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the data
try:
    data = pd.read_csv('expt1.csv')
except FileNotFoundError:
    # Create sample data if file doesn't exist
    np.random.seed(42)
    n_samples = 100
    x1_healthy = np.random.normal(5, 1, n_samples//2)
    x2_healthy = np.random.normal(5, 1, n_samples//2)
    x1_sick = np.random.normal(8, 1, n_samples//2)
    x2_sick = np.random.normal(8, 1, n_samples//2)
    
    data = pd.DataFrame({
        'subjectno': range(n_samples),
        'x1': np.concatenate([x1_healthy, x1_sick]),
        'x2': np.concatenate([x2_healthy, x2_sick]),
        'vss': [0] * (n_samples//2) + [1] * (n_samples//2)
    })

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(data[data['vss'] == 0]['x1'], data[data['vss'] == 0]['x2'], 
            color='red', label='No Vector Space Sickness')
plt.scatter(data[data['vss'] == 1]['x1'], data[data['vss'] == 1]['x2'], 
            color='blue', label='Vector Space Sickness')
plt.xlabel('Subspace Assay (x1)')
plt.ylabel('Basis Balance Scale (x2)')
plt.title('Vector Space Sickness Classification')
plt.legend()
plt.show()

# Prepare data for classification
X = data[['x1', 'x2']]
y = data['vss']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize decision boundary
def plot_decision_boundary(clf, X, y):
    plt.figure(figsize=(10, 6))
    
    # Create a mesh grid
    x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
    y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[y == 0]['x1'], X[y == 0]['x2'], color='red', label='No Vector Space Sickness')
    plt.scatter(X[y == 1]['x1'], X[y == 1]['x2'], color='blue', label='Vector Space Sickness')
    plt.xlabel('Subspace Assay (x1)')
    plt.ylabel('Basis Balance Scale (x2)')
    plt.title('Vector Space Sickness Classification with Decision Boundary')
    plt.legend()
    plt.show()

plot_decision_boundary(clf, X, y)