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
-I was able to generate a classification report, confusion matrix and feature importance graph. All these were plotted on a graph to make inferences.
-For HW2, I uploaded a MNIST dataset and trained a SVM on it. I also loaded and trained a MLP model for logistic regression and MLP Classifier. The training accuracy was 1. The logistic regreesion scored a 92.03%, Random Forest Accuracy was 96.71% and MLP Accuracy was 88.00%.
I set an environment for yolov5. Yolov5 was downloaded from ultralytics and used for predicting the bounding boxes and class probabilities of objects in input images.

## Features

- ✨ HW1 Feature 1: Load dataset and viualize scatter plot
- 🚀 HW1 Feature 2: Train and evaluate model
- 🎯 HW1 Feature 3: Classification report, Confusinon Matrix, Feature Importance graph. 
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

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/assignment-name.git
cd assignment-name
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
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

- Course: [NANO 706 - System and Computational BIOL]
- Instructor: [Prof Sam Oliviera]
- Institution: [North Carolina A & T University]
- Semester: [Fall 2025]

## Contact

**Your Name**
- Email: jkagyenfraasare@aggies.ncat.edu
- GitHub: [https://github.com/Agyenfra]
- LinkedIn: [http.linkedin.com/in/joshua-kofi-agyenfra-asare-716402118]

---

**Note**: This project was created as part of an academic assignment. Please refer to your institution's academic integrity policy before using this code.
