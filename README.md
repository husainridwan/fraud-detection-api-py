# Fraud Detection API

### Introduction
This is an API that is used to show the result of a model which determines if an online transaction is fraudulent or not.

### Description
The dataset was provided by PaySim first paper of the simulator:
E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. The credit card transactions were segmented into different types such as Debit, Transfer, Cash In, Cash Out & Payment. It was observed during exploratory data analysis that majority of fraudulent transactions occurred during Transfer and Debit transactions and mostly with amounts greater than 200000. 

Also, there were errors in the origin and destination balance after the transaction has been completed so new features were created during feature engineering to correct the mistake and they were vital to the classification of transactions as fraudulent or not. Due to the dataset being imbalanced, F1-score was used as the metric for classification and LightGBM and XGBoost Classifier algorithms were used for modeling, then Optuna was used for hyperparameter optimization and the best model gave an F1-score of 0.9977 on test data.

The REST API was developed using FastAPI and was deployment pipeline was created using GitHub Actions which used Docker and Azure App Services.