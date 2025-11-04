# Assignment 2

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This assignment is in 2 folds. The first one labelled as HW1 and the second labelled as HW2. In HW1 I analysed a wine dataset using a model in python. I also predicted and evaluated the model's precison and accuracy. Finally I used python to create a decison tree and a confusion matrix.
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

For HW1 the first part was to load a wine dataet and to visualize the data using a scatter plot.
The second part was to use the dataset to create a model and evaluate the model. 
The dataset for the logistic regression was to build a statistical model to predict the probability of a wine belonging to a specific quality category based on its chemical properties. This task was to classify the wine datasets based on their chemical properties.
After training a data set of 124 with a testing set of 54, I obtained a training accuracy of 0.9839 and a testing acuracy of 1. 
I was also able to generate a Logistic regression with classification report, confusion matrix and feature importance graph. The pictures of the graphs are attached.
For HW2, I uploaded a MNIST dataset and trained a SVM on it. I also loaded and trained a MLP model for logistic regression and MLP Classifier. The training accuracy was 1. The logistic regreesion scored a 92.03%, Random Forest Accuracy was 96.71% and MLP Accuracy was 88.00%. I set an environment for yolov5. Yolov5 was downloaded from ultralytics and used for predicting the bounding boxes and class probabilities of objects in input images. Yolov5 was able to identify my picture as a person and also a car behind me. The picture is attached.

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

## Results

Logistic Regression and Decision Tree:

```bash
Dataset Shape: (178, 13)

Feature Names: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

Target Classes: ['class_0' 'class_1' 'class_2']

First few rows:
   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
0    14.23        1.71  2.43               15.6      127.0           2.80   
1    13.20        1.78  2.14               11.2      100.0           2.65   
2    13.16        2.36  2.67               18.6      101.0           2.80   
3    14.37        1.95  2.50               16.8      113.0           3.85   
4    13.24        2.59  2.87               21.0      118.0           2.80   

   flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
0        3.06                  0.28             2.29             5.64  1.04   
1        2.76                  0.26             1.28             4.38  1.05   
2        3.24                  0.30             2.81             5.68  1.03   
3        3.49                  0.24             2.18             7.80  0.86   
4        2.69                  0.39             1.82             4.32  1.04   

   od280/od315_of_diluted_wines  proline  target  
0                          3.92   1065.0       0  
1                          3.40   1050.0       0  
2                          3.17   1185.0       0  
3                          3.45   1480.0       0  
4                          2.93    735.0       0  

==================================================
LOGISTIC REGRESSION MODEL
==================================================

Accuracy: 0.9814814814814815

Classification Report:
              precision    recall  f1-score   support

     class_0       0.95      1.00      0.97        18
     class_1       1.00      0.95      0.98        21
     class_2       1.00      1.00      1.00        15

    accuracy                           0.98        54
   macro avg       0.98      0.98      0.98        54
weighted avg       0.98      0.98      0.98        54


Confusion Matrix:
[[18  0  0]
 [ 1 20  0]
 [ 0  0 15]]

==================================================
DECISION TREE MODEL
==================================================

Accuracy: 0.9629629629629629

Classification Report:
              precision    recall  f1-score   support

     class_0       1.00      0.94      0.97        18
     class_1       0.91      1.00      0.95        21
     class_2       1.00      0.93      0.97        15

    accuracy                           0.96        54
   macro avg       0.97      0.96      0.96        54
weighted avg       0.97      0.96      0.96        54


Confusion Matrix:
[[17  1  0]
 [ 0 21  0]
 [ 0  1 14]]

Decision tree visualization saved as 'decision_tree.png'
```

Images:
<img width="953" height="637" alt="Snip_TEMP0001" src="https://github.com/user-attachments/assets/5279b974-55b1-4487-99c8-efd517943dbd" />



### HW2 

Train and evaluate logistic model, Yolov5:

```from PIL import Image  # Import the Image module from Pillow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Preprocess the Data
X = X / 255.0  # Normalize pixel values to [0, 1]
y = y.astype(np.uint8)  # Convert labels to integers

#  Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Different Classifiers

# 1. Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

logisticRegression = LogisticRegression().fit(X_train, y_train)
y_pred_logistic = logisticRegression.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)


# --- STEP 0: Prepare sample data (you would use your own data here) ---
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# --- STEP 1: Load the MLP model ---
# Create an instance of the MLPClassifier with desired settings
# (hidden_layer_sizes, activation, solver, etc.)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# --- STEP 2: Fit the model to your training data ---
mlp.fit(X_train, y_train)

# Now your model is trained and ready for predictions
# For example, to print the training accuracy:
print(f"Training accuracy: {mlp.score(X_train, y_train):.2f}")

y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Step 6: Print Accuracy Results
print(f'Logistic Regression: {accuracy_logistic * 100:.2f}%')
print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')
print(f'MLP Accuracy: {accuracy_mlp * 100:.2f}%')

# Set up the environment
!git clone https://github.com/ultralytics/yolov5  # Clone the YOLOv5 repo
%cd yolov5
!pip install -r requirements.txt  # Install requirements

# Import libraries
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5s model (small)

image_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Joshua.jpeg')
image = image_path

# Resize the image
resized_image = image

# Convert the resized image to a format compatible with the model (e.g., NumPy array)
# If your model requires a specific input format, you may need to adjust the image further.
img_array = np.array(resized_image)

# Perform inference
results = model(resized_image)

# Show results
results.show()  # This will display the image with detections

# If you want to visualize the resized image
plt.imshow(resized_image)
plt.axis('off')
plt.title('Resized Image')
plt.show()
```

## Image
<img width="758" height="670" alt="Snip_TEMP0001-2" src="https://github.com/user-attachments/assets/e7a4f1a4-50c7-4d30-bdcd-fd4178063f28" />




## Assignment Requirements

### Completed Requirements

- [x] Requirement 1: Decision tree 
- [x] Requirement 2: Logistic Regression
- [x] Requirement 3: Prediction and Evaluation
- [x] Requirement 4: Yolov5
- [x] Requirement 5: Building Models


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
