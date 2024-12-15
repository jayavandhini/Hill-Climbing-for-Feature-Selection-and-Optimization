# Hill Climbing for Feature Selection and Optimization

## Overview
This project demonstrates the implementation of the **Hill Climbing** algorithm for optimization problems and feature selection tasks. It includes two main applications:

1. **Hill Climbing for Optimization**: A basic hill climbing algorithm that finds the minimum of an objective function (e.g., a quadratic function).
2. **Feature Selection via Hill Climbing**: The hill climbing algorithm is used to select the best subset of features for training a Decision Tree Classifier using a synthetic classification dataset.

## Features

- **Hill Climbing Algorithm**: The main optimization algorithm used in the project is hill climbing, which iteratively improves solutions by evaluating and selecting neighboring candidates based on their cost or score.
- **Objective Function**: The objective functions evaluate the cost or error of solutions. For optimization tasks, the function aims to minimize the result (e.g., `x^2`), and for feature selection, it evaluates model performance based on selected features.
- **Feature Selection**: The project also includes a feature selection component, which evaluates various subsets of features and uses hill climbing to find the best feature subset that maximizes classification performance.
- **Stochastic Search**: The hill climbing algorithm incorporates a stochastic element through random mutations, which introduces variability in the search process.

## Components

1. **Objective Function**: Functions that compute the cost or error of a given solution.
2. **Hill Climbing**: The search algorithm that iteratively explores solutions to optimize the objective.
3. **Feature Selection**: A process that uses hill climbing to select an optimal subset of features for classification tasks.
4. **Data Generation**: The project uses synthetic datasets, generated using `sklearn.datasets.make_classification`, to simulate classification tasks with various feature subsets.
5. **Model Evaluation**: The Decision Tree Classifier from scikit-learn is used for evaluating the performance of selected feature subsets.

## Requirements

The following Python libraries are required for this project:

- `numpy`
- `matplotlib`
- `sklearn`

To install the dependencies, you can use pip:

```bash
pip install numpy matplotlib scikit-learn
```

## How It Works

### Hill Climbing Algorithm

The algorithm starts by generating a random initial solution and evaluating its fitness using the objective function. It then explores neighboring solutions (mutated candidates) and accepts those that improve the current solution. This process continues for a specified number of iterations or until no improvement is found for a given number of iterations (patience).

### Feature Selection with Hill Climbing

In this task, the objective function evaluates the performance of a classifier (Decision Tree) based on a selected subset of features. The hill climbing algorithm iterates over possible subsets of features, mutating and evaluating their performance until it finds the best feature subset.

## How to Run

### 1. Optimization Problem (Hill Climbing)

The optimization part of the project solves a simple quadratic optimization problem. To run the optimization:

```bash
python hill_climbing_optimization.py
```

This will execute the hill climbing algorithm to minimize the function `f(x) = x^2` and display the optimization progress.

### 2. Feature Selection (Hill Climbing)

The feature selection task uses hill climbing to select the best feature subset for a Decision Tree Classifier. To run the feature selection:

```bash
python hill_climbing_feature_selection.py
```

This will execute the hill climbing algorithm for feature selection on a synthetic classification dataset, reporting the best feature subset and its performance.

## Example Output

For the optimization problem, the output will show the progress of the hill climbing algorithm as it iterates over possible solutions:

```
>0 f([0.5]) = 0.25000
>1 f([0.2]) = 0.04000
...
```

For feature selection, the output will display the feature subset selection process:

```
Initial solution: 50 features, Score: 0.8430
Iteration 001: 40 features, Score: 0.8450
Iteration 002: 42 features, Score: 0.8475
...
Done!
Best Solution: 42 features selected, Score: 0.8475
Selected Features: [1, 3, 5, 7, 9, 11, 13, 15, 17]
```
