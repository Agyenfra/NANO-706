# Assignment 2

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This assignment is in 2 folds. The first one labelled as HW1 and the second labelled as HW2. In HW1 I analysed a a wine dataset using a model in python. I also predicted and evaluated the model's precison and accuracy. Finally I used python  create a decison tree and a confusion matrix.
In HW2 I created a model in python, trained the model, analysed the model and set up an enironment to visualize my image using yolov5.


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Assignment Requirements](#assignment-requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

-For HW1 the first part was to load a wine dataet and to visualize the data using a scatter plot.
The second part was to use the dataset to create a model and evaluate the model. After training a data set of 124 with a testing set of 54, I obtained a training accuracy of 0.9839 and a testing acuracy of 1.
-I was able to generate a Logistic regression with classification report, confusion matrix and feature importance graph. All these were plotted on a graph to make inferences.
-For HW2, I uploaded a MNIST dataset and trained a SVM on it. I also loaded and trained a MLP model for logistic regression and MLP Classifier. The training accuracy was 1. The logistic regreesion scored a 92.03%, Random Forest Accuracy was 96.71% and MLP Accuracy was 88.00%.
I set an environment for yolov5. Yolov5 was downloaded from ultralytics and used for predicting the bounding boxes and class probabilities of objects in input images.

## Features

- ✨ HW1 Feature 1: Load dataset and viualize scatter plot
- 🚀 HW1 Feature 2: Prediction and Evaluation
- 🎯 HW1 Feature 3: Logistic Regression 
- 📊 HW2 Feature 4: Train and evaluate logistic model, Yolov5

## Technologies Used

- **Language**: Python 3.9+
- **Libraries**: 
  - NumPy - for numerical computations
  - Pandas - for data manipulation
  - Matplotlib - for data visualization
  - Seaborn - for statistical graphs
  - Scikit-learn (sklearn) - for machine learning
  - Graphviz - for data visualization
  - Pydot - for visualizing data
  - Pytorch - for machine learning libraries
  - CV2 - for openCV library
- **Tools**: Git, Jupyter notebook

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### HW1 

1. Load dataset and viualize scatter plot:
```bash
# Load the Wine Quality dataset
wine = datasets.load_wine()
X = wine.data  # Features
Y = wine.target  # Target variable (wine quality)

# Create a DataFrame for better visualization
wine_df = pd.DataFrame(X, columns=wine.feature_names)
wine_df['target'] = Y

# Display the feature names
print("Feature names:", wine.feature_names)
# Display the first few rows of the dataset
wine_df.head()
# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(wine_df['alcohol'], wine_df['malic_acid'], c=wine_df['target'], cmap='coolwarm', s=30)

plt.title(f"Wine Quality Dataset")
plt.xlabel('alcohol') 
plt.ylabel('malic_acid')
plt.colorbar(ticks=np.unique(Y), label='Quality')
plt.show()

# Standardize features
scaler = StandardScaler()
X2 = scaler.fit_transform(X)

# Split the data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, Y_train)  

Y_pred = svm.predict(X_test)

# Make predictions and calculate accuracy
accuracy = accuracy_score(Y_test , Y_pred)
print(f"Accuracy of the SVM model: {accuracy:.2f}")
```

2. Prediction and Evaluation:
```bash
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {wine.target_names}")
print(f"Feature names: {wine.feature_names}\n")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}\n")

# Create and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}\n")

# Classification report
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=wine.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(cm)

# Feature importance
feature_importance = dt_classifier.feature_importances_
feature_names = wine.feature_names

print("\nFeature Importance:")
for name, importance in sorted(zip(feature_names, feature_importance), 
                               key=lambda x: x[1], reverse=True):
    if importance > 0:
        print(f"{name}: {importance:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Decision Tree visualization
plot_tree(dt_classifier, 
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,
          ax=axes[0, 0],
          fontsize=8)
axes[0, 0].set_title("Decision Tree Structure", fontsize=14, fontweight='bold')

# 2. Confusion Matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names,
            yticklabels=wine.target_names,
            ax=axes[0, 1])
axes[0, 1].set_title("Confusion Matrix", fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. Feature Importance bar chart
sorted_idx = np.argsort(feature_importance)[::-1]
top_n = 10
axes[1, 0].barh(range(top_n), feature_importance[sorted_idx][:top_n])
axes[1, 0].set_yticks(range(top_n))
axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance Score')
axes[1, 0].set_title("Top 10 Feature Importances", fontsize=14, fontweight='bold')

# 4. Accuracy comparison
accuracy_data = [train_accuracy, test_accuracy]
axes[1, 1].bar(['Training', 'Testing'], accuracy_data, color=['#2E86AB', '#A23B72'])
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].set_title("Training vs Testing Accuracy", fontsize=14, fontweight='bold')
for i, v in enumerate(accuracy_data):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nVisualization complete!")
```

3. Logistic Regression:
```bash
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print("Dataset Shape:", X.shape)
print("\nFeature Names:", wine.feature_names)
print("\nTarget Classes:", wine.target_names)
print("\nFirst few rows:")
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale the features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== LOGISTIC REGRESSION ==========
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

# Train logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = log_reg.predict(X_test_scaled)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=wine.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# ========== DECISION TREE ==========
print("\n" + "="*50)
print("DECISION TREE MODEL")
print("="*50)

# Train decision tree model (no scaling needed)
dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_classifier.fit(X_train, y_train)
# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=wine.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# ========== VISUALIZATIONS ==========
# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
print("\nDecision tree visualization saved as 'decision_tree.png'")
plt.show()

# Compare model accuracies
models = ['Logistic Regression', 'Decision Tree']
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_dt)
]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Comparison', fontsize=14)
plt.ylim(0, 1.0)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Model comparison saved as 'model_comparison.png'")
plt.show()

# Feature importance from Decision Tree
feature_importance = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE (Decision Tree)")
print("="*50)
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance from Decision Tree', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")
plt.show()
```

## Usage

Provide clear instructions on how to run your code:

```bash
python main.py
```

Or with command-line arguments:

```bash
python main.py --input data.csv --output results.txt
```

### Configuration

If your project has configuration options, explain them here:

```python
# Example configuration
CONFIG = {
    'data_path': 'data/',
    'output_format': 'json',
    'verbose': True
}
```

## Project Structure

```
assignment-name/
│
├── data/                   # Data files
│   ├── input.csv
│   └── sample_data.json
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # Main entry point
│   ├── utils.py           # Utility functions
│   └── models.py          # Data models/classes
│
├── tests/                  # Unit tests
│   ├── test_main.py
│   └── test_utils.py
│
├── docs/                   # Documentation
│   └── report.pdf
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── LICENSE                # License file
```

## Examples

### Example 1: Basic Usage

```python
from src.main import process_data

# Load and process data
result = process_data('data/input.csv')
print(f"Processing complete: {result}")
```

**Output:**
```
Processing complete: 100 records processed successfully
```

### Example 2: Advanced Usage

```python
from src.models import DataProcessor

# Create processor with custom parameters
processor = DataProcessor(threshold=0.5, verbose=True)
processor.run()
```

## Assignment Requirements

### Completed Requirements

- [x] Requirement 1: Implement data loading functionality
- [x] Requirement 2: Create data processing algorithm
- [x] Requirement 3: Generate visualizations
- [x] Requirement 4: Write unit tests
- [x] Requirement 5: Document code with comments

### Implementation Details

**Algorithm Complexity**: O(n log n)

**Key Design Decisions**:
- Used object-oriented approach for better modularity
- Implemented error handling for robust execution
- Followed PEP 8 style guidelines

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or for verbose output:

```bash
python -m pytest tests/ -v
```

## Known Issues

- Issue 1: Defining models can be a challenge. Make sure models are well defined.
- Issue 2: Libraries need to be run from the top for some codes to work.

## Future Improvements

- [ ] Add support for additional data formats
- [ ] Implement parallel processing
- [ ] Create web interface
- [ ] Improve error messages

## Contributing

This is an academic assignment, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Course: NANO 706 - System and Computational BIOL
- Instructor: Prof Sam Oliviera
- Institution: North Carolina A & T University
- Semester: Fall 2025

## Contact

**Your Name**
- Email: jkagyenfraasare@aggies.ncat.edu
- GitHub: https://github.com/Agyenfra
- LinkedIn: http://linkedin.com/in/joshua-kofi-agyenfra-asare-716402118

---

**Note**: This project was created as part of an academic assignment. Please refer to your institution's academic integrity policy before using this code.
