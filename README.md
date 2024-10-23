# Google Play Store - App Rating Prediction

This project aims to predict the ratings of applications on the Google Play Store using machine learning techniques. The dataset, obtained from [Kaggle](https://www.kaggle.com/lava18/google-play-store-apps), contains information about various apps, including their category, number of reviews, size, install count, type (free or paid), price, content rating, genre, last update, current version, and Android version requirements. Predicting app ratings can provide valuable insights for developers to improve their apps and better cater to their target audience.

## Table of Contents
- [Dataset Details](#dataset-details)
- [Project Workflow](#project-workflow)
  - [Data Gathering](#data-gathering)
  - [Data Assessing](#data-assessing)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Model Prediction](#model-prediction)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [References](#references)

## Dataset Details
The Google Play Store dataset contains information about 10,841 apps and consists of 13 columns:

1. **App**: Name of the app
2. **Category**: App category (e.g., "Tools", "Games")
3. **Rating**: User rating (Target variable)
4. **Reviews**: Number of user reviews
5. **Size**: Size of the app (in MB)
6. **Installs**: Number of times the app has been installed
7. **Type**: Type of the app (Free/Paid)
8. **Price**: Price of the app (in USD)
9. **Content Rating**: Suitable age group for the app (e.g., "Everyone", "Teen")
10. **Genres**: Detailed category of the app
11. **Last Updated**: Last update date of the app
12. **Current Version**: Current version of the app
13. **Android Version**: Minimum Android version required to run the app

The target variable is the **Rating**, which we aim to predict based on the other features.

## Project Workflow

### 1. Data Gathering
The data was obtained from [Kaggle's Google Play Store Apps dataset](https://www.kaggle.com/lava18/google-play-store-apps). This dataset contains web-scraped information about various apps on the Play Store.

### 2. Data Assessing
The data was assessed for:
- **Missing values**: Identifying columns with missing data (e.g., "Rating" and "Size").
- **Data types**: Ensuring the correct data types were used for each column.
- **Inconsistencies or anomalies**: Checking for outliers or incorrect data entries.

### 3. Data Cleaning
Data cleaning steps included:
- **Handling missing values**: Imputing or removing missing values in columns such as "Rating" and "Size".
- **Correcting data types**: Converting features like "Reviews" to numeric types and parsing dates for "Last Updated".
- **Removing duplicates**: Ensuring no duplicate entries existed in the dataset.
- **Encoding categorical variables**: Encoding features like "Category", "Type", and "Content Rating".

### 4. Exploratory Data Analysis (EDA)
The EDA phase involved analyzing the distributions of features, relationships between variables, and identifying patterns. Some key findings:
- **Number of reviews and installs** were strong indicators of an app's popularity.
- **Free vs. paid apps** showed different rating distributions.
- **Category and content rating** had varying effects on the average app rating.

### 5. Feature Engineering
Feature engineering steps included:
- **Creating new features**: For example, calculating the number of days since the last update.
- **Transforming features**: Normalizing numerical features like "Size" and "Reviews" for better model performance.
- **One-hot encoding**: For categorical features like "Category" and "Content Rating".

### 6. Model Prediction
Three regression models were used to predict app ratings:

#### 1. Linear Regression
A basic approach where the relationship between input features and the target variable is assumed to be linear.
```python
pipe = make_pipeline(column_trans, linreg)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```
- **MAE**: 0.3489
- **MSE**: 0.2584
- **RMSE**: 0.5907

#### 2. Support Vector Regressor (SVR)
A more complex approach that finds a hyperplane that best fits the data, allowing some error margin.
```python
pipe = make_pipeline(column_trans, svr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```
- **MAE**: 0.3353
- **MSE**: 0.2711
- **RMSE**: 0.5791

#### 3. Random Forest Regressor
An ensemble approach that uses multiple decision trees to improve prediction accuracy by averaging results.
```python
pipe = make_pipeline(column_trans, forest_model)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```
- **MAE**: 0.3473
- **MSE**: 0.2566
- **RMSE**: 0.5893

## Evaluation Metrics
To evaluate model performance, the following metrics were used:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared errors, providing insight into prediction accuracy.

## Results
The **Support Vector Regressor (SVR)** achieved the lowest error values:
- **SVR MAE**: 0.3353
- **SVR MSE**: 0.2711
- **SVR RMSE**: 0.5791

## Future Improvements
- **Hyperparameter Tuning**: Optimize model parameters using techniques like GridSearchCV.
- **Advanced Feature Engineering**: Incorporate more meaningful features, such as interaction terms or clustering information.
- **Neural Networks or Deep Learning Approaches**: Use deep learning methods like neural networks for potential improvements.

## Conclusion
Predicting app ratings on the Google Play Store can help developers understand key factors influencing app success. The **Support Vector Regressor (SVR)** outperformed other models, though further improvements could be achieved through hyperparameter tuning and advanced techniques.

## References
- [Kaggle Dataset - Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps)

---
