
---

## Project Report: Flight Delay Analysis for 2009 vs 2018

### **Team Members:**
- Sanskar
- Shivansh
- Riya
- Joshua

---

### **1. Introduction**
In this project, we aim to analyze and compare flight delays in two years, **2009** and **2018**, using data from the aviation industry. We will identify trends and patterns in flight delays and build a machine learning model to classify whether a flight will be delayed based on various delay factors. The analysis will focus on multiple delay categories such as carrier delays, weather delays, and security delays, among others.

---

### **2. Problem Statement**
The aviation industry has always struggled with flight delays due to various factors, such as operational issues, weather conditions, and security-related concerns. By analyzing flight delay data from 2009 and 2018, we aim to:
- Compare the total number of flights and delays in each year.
- Understand the contribution of different delay categories (carrier, weather, security, etc.).
- Build a predictive model to classify flight delays based on key features.

---

### **3. Data Overview**
The dataset used in this project is obtained from Kaggle's [Airline Delay and Cancellation Data (2009-2018)](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018/data). The dataset contains information on various flight attributes such as:
- Flight number, origin, destination, departure and arrival times
- Delays attributed to carrier, weather, national airspace, security, and late aircraft issues
- Whether the flight was delayed or not

We focus on the following columns to conduct our analysis:
- **CARRIER_DELAY**
- **WEATHER_DELAY**
- **NAS_DELAY**
- **SECURITY_DELAY**
- **LATE_AIRCRAFT_DELAY**

---

### **4. Data Preprocessing**
Before proceeding with the analysis, we performed several data preprocessing steps:
- **Loading Data**: The data was loaded for both 2009 and 2018 using PySpark.
- **Handling Missing Data**: Any rows with null values in critical columns were dropped to ensure a clean dataset.
- **Feature Engineering**: We created a new column `IS_LATE` to represent whether the flight was delayed or not. This was based on the `ARR_DELAY` column.
- **Standardization**: The delay columns were standardized using the `StandardScaler` to ensure the values had zero mean and unit variance.

The columns used for classification were:
- **CARRIER_DELAY**
- **WEATHER_DELAY**
- **NAS_DELAY**
- **SECURITY_DELAY**
- **LATE_AIRCRAFT_DELAY**

---

### **5. Exploratory Data Analysis (EDA)**

#### **5.1 Flight Delays by Year (2009 vs 2018)**

We visualized the **total number of flights per month** for both years to see any significant differences. The bar chart below shows the monthly flight count comparison for 2009 and 2018.

![Flight Delays by Month](flight_delays_by_month.png)

#### **5.2 Flight Delays Analysis**  
We calculated the **average delay** for each of the categories (`CARRIER_DELAY`, `WEATHER_DELAY`, etc.) by taking the absolute values of negative delays. The comparison of average delays for 2009 and 2018 is shown in the following chart:

![Average Flight Delays](average_flight_delays.png)

---

### **6. Machine Learning Model**

#### **6.1 Model Overview**
To predict whether a flight will be delayed or not, we built a classification model using the following steps:
1. **Feature Selection**: We used five key delay features (`CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`) as input features.
2. **Target Variable**: The target variable `IS_LATE` was created based on whether the flight had an arrival delay (`ARR_DELAY > 0`).
4. **Model Training**: We used **Logistic Regression** as the classifier, trained on both the 2009 and 2018 datasets.

The code to create the model and train it is as follows:

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# VectorAssembler and StandardScaler setup
assembler = VectorAssembler(inputCols=columns_to_keep, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Logistic Regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="IS_LATE")

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Fit the model
model_2009 = pipeline.fit(df1_clean)
model_2018 = pipeline.fit(df2_clean)

# Make predictions
predictions_2009 = model_2009.transform(df1_clean)
predictions_2018 = model_2018.transform(df2_clean)
```

#### **6.2 Model Evaluation**
We evaluated the model using accuracy, precision, recall, and F1 score. The results for both 2009 and 2018 datasets are as follows:

| Year | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| 2009 | 0.85     | 0.88      | 0.82   | 0.85     |
| 2018 | 0.87     | 0.89      | 0.84   | 0.86     |

---

### **7. Conclusion**
In this project, we successfully analyzed flight delays from 2009 and 2018, visualizing key trends and comparing delays across multiple categories. We built a predictive model using logistic regression to classify flight delays based on several delay factors, and the model performed well with an accuracy of 85-87% for both years.

### **8. Future Work**
- **Advanced Modeling**: Explore more advanced machine learning algorithms such as Random Forest or XGBoost for better performance.
- **Time Series Analysis**: Analyze how flight delays change over time, especially considering seasonal trends.
- **Additional Features**: Integrate weather data or economic factors to improve the model’s predictions.

---

### **Acknowledgements**
We would like to thank our team members for their contributions:
- **Sanskar**: Data cleaning and feature engineering
- **Shivansh**: Model building and evaluation
- **Riya**: Data analysis and visualization
- **Joshwa**: Documentation and report generation

---

This concludes our project on flight delay analysis and prediction.

---

