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
# This is the code for loading the Wine Quality dataset and creating a dataframe for visualization
```
<img width="953" height="201" alt="Snip_TEMP0001-8" src="https://github.com/user-attachments/assets/ddccba29-5e72-42ce-a0d3-eecb83623792" />

```bash
# This is the code for ploting the scatter plot
```
<img width="953" height="207" alt="Snip_TEMP0001-9" src="https://github.com/user-attachments/assets/69d339c5-89d4-4a64-a3d7-325bc37fc1e8" />



2. Prediction and Evaluation:

```bash
# This is the code for training the dataset and making the predictions
```

<img width="953" height="430" alt="Snip_TEMP0001-10" src="https://github.com/user-attachments/assets/ac44d1bd-40e4-4c51-89cd-88fafc037bba" />

```bash
# This is the code for training the dataset and making the Decision tree
```

<img width="953" height="377" alt="Snip_TEMP0001-11" src="https://github.com/user-attachments/assets/d04c2235-ef36-4e87-8b84-2bf6e39b55b7" />

```bash
# This is the code to make predictions and evaluate the model
```
<img width="953" height="282" alt="Snip_TEMP0001-12" src="https://github.com/user-attachments/assets/974f15ad-cc62-4091-9366-9400313322ac" />


```bash
# This is the code to Visualization of the decision tree, confusion matrix and accuracy comparison
```

<img width="921" height="670" alt="Snip_TEMP0001-13" src="https://github.com/user-attachments/assets/1ef16646-2c1f-49f3-a438-81dbb1357e87" />



3. Logistic Regression:


```bash
# This is the code to Train data set for Logistic regression
```
<img width="953" height="635" alt="Snip_TEMP0001-14" src="https://github.com/user-attachments/assets/2e1a06f5-518f-4e5f-999d-0c5caea14614" />




 

Images:

<img width="953" height="637" alt="Snip_TEMP0001" src="https://github.com/user-attachments/assets/5279b974-55b1-4487-99c8-efd517943dbd" />

<img width="875" height="670" alt="Snip_TEMP0001-3" src="https://github.com/user-attachments/assets/95cb3478-b0c0-4d4f-b283-a5c58c0cd44c" />

<img width="775" height="670" alt="Snip_TEMP0001-5" src="https://github.com/user-attachments/assets/adffdb56-dac8-42e3-8f7d-e6ae667b941a" />

<img width="764" height="670" alt="Snip_TEMP0001-6" src="https://github.com/user-attachments/assets/7ba7f5e6-c38d-4e6d-a38c-45b2f675dbfe" />

<img width="953" height="641" alt="Snip_TEMP0001-7" src="https://github.com/user-attachments/assets/40e93aad-d0a1-4454-83fa-30ca1d700396" />



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
