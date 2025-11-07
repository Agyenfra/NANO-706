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
- [Images](#images)
- [Assignment Requirements](#assignment-requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


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

For HW1 the first part was to load a wine dataet and to visualize the data using a scatter plot.


<img width="953" height="637" alt="Snip_TEMP0001" src="https://github.com/user-attachments/assets/5279b974-55b1-4487-99c8-efd517943dbd" />



2. Prediction and Evaluation:

```bash

The second part was to use the dataset to create a model and evaluate the model. 
The dataset for the logistic regression was to build a statistical model to predict the probability of a
wine belonging to a specific quality category based on its chemical properties.
This task was to classify the wine datasets based on their chemical properties.
After training a data set of 124 with a testing set of 54, I obtained a training accuracy of 0.9839
and a testing acuracy of 1. 

```

<img width="875" height="670" alt="Snip_TEMP0001-3" src="https://github.com/user-attachments/assets/95cb3478-b0c0-4d4f-b283-a5c58c0cd44c" />


3. Logistic Regression:


```bash
# I was also able to generate a Logistic regression with classification report, confusion matrix and feature importance graph.

```
 

Images:

<img width="775" height="670" alt="Snip_TEMP0001-5" src="https://github.com/user-attachments/assets/adffdb56-dac8-42e3-8f7d-e6ae667b941a" />

<img width="764" height="670" alt="Snip_TEMP0001-6" src="https://github.com/user-attachments/assets/7ba7f5e6-c38d-4e6d-a38c-45b2f675dbfe" />

<img width="953" height="641" alt="Snip_TEMP0001-7" src="https://github.com/user-attachments/assets/40e93aad-d0a1-4454-83fa-30ca1d700396" />



### HW2 

I uploaded a MNIST dataset and trained a SVM on it. 
I also loaded and trained a MLP model for logistic regression and MLP Classifier. 
The training accuracy was 1. 
The logistic regreesion scored a 92.03%, Random Forest Accuracy was 96.71% and MLP Accuracy was 88.00%. 
I set an environment for yolov5. 
Yolov5 was downloaded from ultralytics and used for predicting the bounding boxes 
and class probabilities of objects in input images. 
Yolov5 was able to identify my picture as a person and also a car behind me. 


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
