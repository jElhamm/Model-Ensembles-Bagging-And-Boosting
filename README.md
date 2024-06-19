# Model Ensembles & Boosting in Machine Learning

   Welcome to the **Model-Ensembles-Boosting-In-Machine-Learning** repository. 
   This project demonstrates the implementation of various boosting methods using the *Titanic Dataset*, providing a comprehensive comparison of their performances.

## Table of Contents

   1. [Introduction](#introduction)
   2. [Dataset](#dataset)
   3. [Boosting Methods](#boosting-methods)
       - [Decision Tree](#boosting-with-decision-tree)
       - [XGBoost](#boosting-with-xgboost)
       - [LightGBM](#boosting-with-lightgbm)
   4. [Installation](#installation)
   5. [Usage](#usage)
   6. [Example Results](#results)
   7. [License](#license)

## Introduction

   Boosting is a powerful ensemble technique in machine learning that combines the predictions of several base estimators to improve robustness over a single estimator. 
   
   
   This repository explores:

   - [Boosting with XGBoost](Boosting%20With%20XGboost)
   - [Boosting with LightGBM](Boosting%20With%20lightgbm)
   - [Boosting with Decision Tree](Boosting%20With%20DecisionTree)


   Each method is tested on the Titanic dataset to illustrate their effectiveness and performance differences.

## Dataset

   The [Titanic dataset](titanic.csv) is a famous dataset used in machine learning and statistics for binary classification.
   The goal is to predict whether a passenger survived the Titanic disaster based on features such as age, gender, class, etc.

## Boosting Methods

   * Boosting with Decision Tree

   [Decision Trees](Boosting%20With%20DecisionTree/Boosting_Implement_With_DecisionTree.ipynb) are simple yet powerful models. Here, we implement boosting with decision trees to enhance their performance by iteratively correcting the errors of the previous models.

   * Boosting with XGBoost

   [XGBoost](Boosting%20With%20XGboost/Boosting_Implement_With_XGboost.ipynb) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

   * Boosting with LightGBM

   [LightGBM](Boosting%20With%20lightgbm/Boosting_Implement_With_lightgbm.ipynb) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages:
   - Faster training speed and higher efficiency
   - Lower memory usage
   - Better accuracy

## Installation

   To run the scripts and notebooks in this repository, ensure you have Python 3.x and the following libraries installed:

   | Library      | Version |
   |--------------|---------|
   | numpy        | 1.21.0  |
   | pandas       | 1.3.0   |
   | matplotlib   | 3.4.2   |
   | scikit-learn | 0.24.2  |
   | lightgbm     | 3.2.1   |
   | xgboost      | 1.4.0   |


   You can install these dependencies using pip:

   'pip install numpy pandas matplotlib scikit-learn lightgbm xgboost'

## Usage

   1. Clone the repository
   2. Open and run the Jupyter notebooks (*.ipynb) in your preferred environment. Each notebook demonstrates a specific boosting method.

## Example Results

   Outputs and obtained results are only used for the dataset. If it uses another dataset, you will get other outputs.

   1. Classification Report Comparison

| Algorithm   | Precision (Not Survived) | Recall (Not Survived) | F1-score (Not Survived) | Precision (Survived) | Recall (Survived) | F1-score (Survived) | Accuracy |
|-------------|--------------------------|-----------------------|--------------------------|----------------------|-------------------|---------------------|----------|
| **LightGBM** | 0.75                     | 0.64                  | 0.69                     | 0.80                 | 0.87              | 0.83                | 0.78     |
| **XGBoost**  | 0.73                     | 0.57                  | 0.64                     | 0.77                 | 0.87              | 0.82                | 0.76     |
| **AdaBoost** | 0.00                     | 0.00                  | 0.00                     | 0.61                 | 0.96              | 0.75                | 0.59     |

   - **Accuracy:** Overall accuracy of the model on the test set.
   - **Precision:** Ability of the model to avoid false positives.
   - **Recall:** Ability of the model to find all positive instances.
   - **F1-score:** Harmonic mean of precision and recall.

   Based on the evaluation of the classification reports and comparison among the algorithms, we observe varying performance across different metrics. Each algorithm shows strengths in different aspects of the classification task, with LightGBM and XGBoost generally outperforming AdaBoost on this dataset.

   2.  ROC

   * The [Receiver Operating Characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
   The curve is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
   * The area under the ROC curve (AUC) provides an aggregate measure of performance across all classification thresholds.
   An AUC of 1.0 represents a perfect model, while an AUC of 0.5 represents a model with no discrimination ability.


   3. Confusion Matrix

   * The [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix) is a table used to describe the performance of a classification model on a set of test data for which the true values are known.
   It shows the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions.
   * This matrix helps in understanding the types of errors the model is making and provides insights into the accuracy, precision, recall, and F1-score metrics.

## License

   This repository is licensed under the GNU General Public License (GPL) v3.0.
   See the [LICENSE](./LICENSE) file for more details.