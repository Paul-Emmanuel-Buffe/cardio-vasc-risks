# Cardiovascular Risk Prediction: Logistic Regression Analysis

## Project Overview
This project is developed within the framework of **preventive medicine**. Its primary objective is to design a diagnostic tool capable of predicting cardiovascular risks using a binary classification approach. 

In France, cardiovascular diseases represent the second leading cause of mortality. Between 300,000 and 400,000 cardiovascular accidents occur annually, one-third of which are fatal. This tool leverages machine learning to assist in early detection and provide lifestyle recommendations to patients.

## Technical Context: Logistic Regression
The core of this study relies on **Logistic Regression**, a statistical method used to predict a binary dependent variable (values such as 0/1, True/False, or Yes/No) based on quantitative explanatory variables. Unlike linear regression, it focuses on classification rather than continuous value prediction.

## Risk Factor Analysis
The model analyzes a network of 12 cardiovascular risk factors, categorized into four distinct groups based on their interactions:

1. **Non-modifiable factors**: Sex, age, and family history. These predict other factors but cannot be modified by the patient.
2. **Lifestyle factors**: Smoking, sedentary behavior, and alcohol abuse. These predict many other factors but are rarely predicted by them.
3. **Upstream clinical factors**: Sleep disorders, obesity, and depression. These both predict and are predicted by numerous factors.
4. **Downstream clinical factors**: Hypertension, dyslipidemia, and diabetes. These are predicted by many factors but predict very few themselves.

## Methodology and Evaluation
The development process follows a structured pipeline:
* **Data Preprocessing**: Handling missing values, outliers, and duplicates to ensure medical consistency.
* **Exploratory Data Analysis (EDA)**: Visualizing factor interactions using Matplotlib, Seaborn, or Plotly.
* **Modeling**: 
    * Implementation using **Scikit-Learn** with hyperparameter tuning.
    * Development of a **custom Python class** for logistic regression without external ML libraries.
* **Performance Metrics**: Evaluation using a confusion matrix, accuracy, recall, and classification reports.

## Repository Structure
* `notebooks/exploration.ipynb`: Data cleaning, analysis, and visualization.
* `notebooks/modelisation.ipynb`: Model training, custom class implementation, and performance evaluation.
* `data/`: Patient datasets collected through medical partnerships.

## Conclusion
The final model provides a binary diagnostic output to determine if a subject is at risk. This includes a specific case study analysis for a 53-year-old male subject ("Arthur") to validate the model's predictive capabilities in a real-world scenario.