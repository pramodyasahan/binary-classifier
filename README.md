# Customer Churn Prediction Model

## Overview
This repository houses the code for a machine learning model designed to predict customer churn. The model is built using Support Vector Machine (SVM) from the scikit-learn library and incorporates preprocessing, pipeline, and grid search techniques for optimal performance.

## Data
The model utilizes two datasets for training and testing:
- `train 2.csv`: Contains training data with customer features and churn status.
- `test 2.csv`: Contains testing data with customer features.

## Features
The datasets comprise both numerical and categorical customer-related features. These features undergo preprocessing to make them suitable for feeding into the model.

## Preprocessing
Key preprocessing steps include:
- Scaling numerical features using StandardScaler.
- Encoding categorical features using OneHotEncoder.
- Removal of non-predictive features like 'id', 'CustomerId', and 'Surname'.

## Model Training
The SVM model is trained on preprocessed data. Key aspects of the model training include:
- Utilizing the SVC (Support Vector Classification) algorithm.
- Employing GridSearchCV for hyperparameter tuning.

## Hyperparameters
The following hyperparameters are considered in the grid search:
- 'C': [0.1, 1] (Regularization parameter)
- 'gamma': [1, 0.1] (Kernel coefficient)
- 'kernel': ['rbf', 'poly', 'sigmoid'] (Specifies the kernel type)

## Usage
To use the model:
1. Load the training and testing datasets.
2. Preprocess the datasets by scaling numerical features and encoding categorical features.
3. Train the SVC model using the training data.
4. Predict churn status for the test data.
5. Export the predictions to 'predictions.csv'.

## Dependencies
- numpy
- pandas
- scikit-learn

## Note
This code serves as a basic structure for customer churn prediction and can be further optimized for specific use-cases and datasets.
