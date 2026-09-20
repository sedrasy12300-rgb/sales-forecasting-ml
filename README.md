 📊 Sales Forecasting
 
📌 Project Overview

This project focuses on predicting future sales using historical sales data, store information, product families, and time-based features.

The project applies Machine Learning for time-series forecasting, with feature engineering techniques based on historical sales patterns.


🎯 Project Goal

The main goal is to build a regression model that can predict future daily sales based on:

Historical sales

Store information

Product family

Promotional activity

Time-based features

Previous sales patterns


📂 Dataset

The project uses sales data along with store information.

Sales Data

The main features include:

date — Sales date

store_nbr — Store number

family — Product family

sales — Sales value

onpromotion — Number of products on promotion

Store Data

Additional store information includes:

city

state

type

cluster

The datasets are merged using store_nbr.

🧹 Data Preprocessing
The following preprocessing steps were applied:
Converted the date column to datetime format.
Extracted time-based features:
Year
Month
Day
Day of week
Merged sales data with store information.
Handled a specific abnormal sales value.
Sorted the data by store, product family, and date.
Created historical sales features.
Removed rows with missing lag values.

⚙️ Feature Engineering

Historical sales information was used to create additional features.
Lag Features
sales_lag1 — Previous day's sales
sales_lag7 — Sales from 7 days earlier
sales_lag30 — Sales from 30 days earlier
Rolling Features
sales_rolling7 — Average sales over the previous 7 days
sales_rolling30 — Average sales over the previous 30 days
These features help the model capture recent and longer-term sales patterns.

🔄 Data Preparation

Categorical features were encoded using One-Hot Encoding with OneHotEncoder.
The preprocessing pipeline was implemented using ColumnTransformer.
The data was split chronologically rather than randomly to preserve the time order of the dataset.
The split date was:
2016-09-18
Approximately 80% of the data was used for training and the remaining data was used for testing.

🤖 Machine Learning Model

The model used in this project is:
Random Forest Regressor
Model Parameters
n_estimators = 100
max_depth = 15
random_state = 42
n_jobs = -1
Due to the large size of the dataset, a subset of the training data was used when fitting the model.

📊 Model Evaluation

The model was evaluated using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
Results

The model achieved an R² score of approximately 0.914 on the test set.
R² is used here as a regression evaluation metric and should not be interpreted as classification accuracy.

🔎 Error Analysis

The prediction errors were also analyzed using several statistics.

The largest error was caused by an unusually high sales value for a specific store and product family, showing the effect that extreme observations can have on sales forecasting models.

🔮 Future Sales Prediction

A separate prediction process was created to forecast sales from:
2017-08-16 to 2017-08-31
Because the model uses historical sales features, predictions are generated sequentially.
The prediction for one day is added to the historical data and can then be used as a feature when predicting the following day.
This creates a simple recursive forecasting process.

💾 Saved Models

The trained components are saved using Joblib:
model.pkl — Trained Random Forest model
preprocessor.pkl — Data preprocessing pipeline
This allows the trained model to be reused without retraining it from scratch.

📄 Output
The prediction process generates:
subimission3.csv
The file contains:
id
sales
where sales represents the predicted sales value.

🗂️ Project Structure

Sales-Forecasting/
│

├── customer_sales_prediction.py

├── prediction.py

├── model.pkl

├── preprocessor.pkl

├── subimission3.csv

├── README.md

└── requirements.txt

🛠️ Technologies Used

Python
Pandas
Scikit-learn
Random Forest Regressor
One-Hot Encoding
ColumnTransformer
Joblib
Matplotlib
📚 What I Learned

Through this project, I practiced:
Working with large datasets
Time-based data preprocessing
Merging multiple datasets
Feature engineering for time-series data
Creating lag and rolling features
Building regression models
Evaluating regression performance
Analyzing prediction errors
Handling extreme values
Saving and reusing trained ML models
Generating recursive future predictions

👩‍💻 Author

Sedra Abdulhamid Marei
Computer Engineering Student

Machine Learning & Data Science
GitHub: https://github.com/sedrasy12300-rgb 
