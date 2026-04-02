# Gold Price Forecasting using LSTM(Long Short-Term Memory)
![images (1)](https://github.com/user-attachments/assets/5d71067b-3528-4b63-ac9f-3f53dbdf9481)

# Problem Statement
The prediction of gold prices is meaningful for multiple industries. Gold is a relatively stable store of value, and its price is closely linked to stocks, exchange rates, and monetary policy. Observing the trend of gold prices can promote the stable development of financial businesses. In this project, the daily opening and closing prices of gold of ten years were chosen as the data support. 
In this notebook, we are going to build a Long short term memory model to predict the future price of gold, which can be very useful for traders. For this purpose, we use the historical gold price data for 10 years (from 2013 to 2023).

# Objectives-
•	Exploring and Preprocessing Data: Analyzing the data and readying it for training and testing phases.
•	Model Building: Employing LSTM (Long Short Term Memory) to predict the gold price.
•	Prediction and Analysis: Forecasting gold prices for the year 2022 and analyzing the model's performance.

# Dataset-
This dataset offers a detailed look into gold price trends from 2013 to 2022. Each row pertains to a specific date, and they're all organized in chronological order.
<img width="820" height="341" alt="image" src="https://github.com/user-attachments/assets/dc9d74d9-3cba-43b2-92d0-1bc6fd00d3af" />

# Exploring and Preprocessing of Data
Data preprocessing is the first step in any data analysis or machine learning pipeline. It involves cleaning, transforming and organizing raw data to ensure it is accurate, consistent and ready for modeling. It has a big impact on model building such as:
•	Clean and well-structured data allows models to learn meaningful patterns rather than noise.
•	Properly processed data prevents misleading inputs, leading to more reliable predictions.
•	Organized data makes it simpler to create useful inputs for the model, enhancing model performance.
•	Organized data supports better Exploratory Data Analysis (EDA), making patterns and trends more interpretable.

Gold_Price_2013-2023.csv: CSV file containing the entire dataset.
Data overview
<img width="555" height="367" alt="image" src="https://github.com/user-attachments/assets/99517b65-ee1a-42f2-9471-5e413e7051e6" />
•	This dataset contains gold price data from year 2013 to 2022.
•	Daily gold price information including daily open, high and low prices and the final price of each day (price) along with the volume of transactions and price changes in each day.
•	Shape of the dataset is 2583 rows × 7 columns.
•	All features are stored as object datatype.
•	There are some missing values in vol feature
Steps used in this project Data Processing
1.	We have Data feature in object dtype, we have to convert that into datetime format using pd.to_datetime constructor.
2.	The " , " sign is redundant in the dataset. First, we remove it from the entire dataset. 
3.	Change the data type of the numerical variables to float.
4.	Checking for null values and getting rid of null values
5.	Sorting dates in ascending order
6.	Setting date feature as index of dataset

# Exploratory Data Analysis
Exploratory Data Analysis (EDA) is an important step in data science and data analytics as it visualizes data to understand its main 
features, find patterns and discover how different parts of the data are connected.

## 1.	Visualization of Gold price over time(2013-2023)

 <img width="1004" height="541" alt="image" src="https://github.com/user-attachments/assets/c1880e0c-3066-4a9d-8718-e6bf5e4c0254" />
 
•	From the above visualization, we can see there is lot of randomness present in the data, because, here data volume is in high quantity.
•	To imporve the readability and understand the overall trends in data, we will use "Moving Average Method" to visualize this data.


# "Moving Average Method" – 
The Moving Average (MA) method is used to visualize time-series data by smoothing out short-term, random fluctuations (noise) to highlight underlying trends or cycles. It calculates the average of a fixed subset of data points, moving this window forward to constantly update the trend line.
rolling_mean = gold_df['Price'].rolling(window=30).mean(): This creates a new pandas series that calculates the average price of the last 30 days for every data point in the 'Price' column.

## 2.	Visualization of Gold price over time (2013-2023) Using "Moving Average Method"

 <img width="1010" height="573" alt="image" src="https://github.com/user-attachments/assets/608afcba-a097-4a39-bcee-05c0507ec9f1" />
 
From this visualization -
•	from 2013 to 2016 we can see that gold price was continuosly falling down with minor upword trends.
•	Again from 2016 to untill 2019 we can able to see mix trend.
•	But after 2019 gold prices has shown strong long term upward trend.
•	In the last few months of 2023, the gold market has experienced a mild upward trend


## 3.	Distribution of Gold price – Histogram
A histogram is a graph showing frequency distributions. It is a graph showing the number of observations within each given interval.
sns.histplot(gold_df['Price'], bins = 50, kde = True, color = 'orange')

 <img width="981" height="379" alt="image" src="https://github.com/user-attachments/assets/09a2d5ef-df6e-42b9-a97b-67367d4b6857" />
 
From this plot-
•	Gold price stayed between 1200 to 1400 for longest time.
•	Then second long stay is around 1800.


## 4.	The Autocorrelation Function (ACF)
The ACF plots the correlation of the time series with itself at different lags. This helps in identifying patterns such as seasonality, trends, and the persistence of values over time.   
Key Concepts of ACF
•	Correlation with Lags: ACF quantifies the linear relationship between a data point and previous points (e.g., today's value vs. yesterday's, last week's, etc.).
•	ACF Plot (Correlogram):  A graph showing the correlation coefficient on the y-axis and the time lag on the x-axis, making patterns visually clear.
•	Interpreting Patterns:
•	Slow Decay: Indicates trends or non-stationarity.
•	Periodic Peaks: Suggests seasonality.
•	Quick Dampening to Zero: Points to randomness or white noise.

<img width="1063" height="479" alt="image" src="https://github.com/user-attachments/assets/acc94ef9-bad6-42b1-a2a9-1288b11daf06" />

Interpretation-
•	Slow Decrease in ACF, Suggests a trend is present, indicating non-stationary data.


## 5.	Seasonal Decomposition
Time series Seasonal  decomposition is the process of separating a time series into its constituent components, such as trend, seasonality, and noise.
Time series decomposition helps us break down a time series dataset into three main components:
•	Trend: The trend component represents the long-term movement in the data, representing the underlying pattern.
•	Seasonality: The seasonality component represents the repeating, short-term fluctuations caused by factors like seasons or cycles.
•	Residual (Noise): The residual component represents random variability that remains after removing the trend and seasonality.

By separating these components, we can gain insights into the behavior of the data and make better forecasts.

<img width="964" height="536" alt="image" src="https://github.com/user-attachments/assets/95ae8644-46b9-497a-9bec-bb82dc40056a" />


## 6.	Daily Change% in price (Volatility)
The percentage change in the gold price compared to the previous day
 <img width="1004" height="468" alt="image" src="https://github.com/user-attachments/assets/db8f6214-4641-4a88-8916-e126435c27a4" />
There are few times, where price has changes very abruptly.


## 7.	Box plots
Box plots are used to visually identify outliers, which are typically displayed as individual points or "fliers" beyond the "whiskers" of the plot. Common libraries like Matplotlib, Seaborn, and Plotly provide functions to create these plots and interact with the outlier data. 
<img width="976" height="532" alt="image" src="https://github.com/user-attachments/assets/6dbdc943-32b3-43b8-92a7-4ad28b2f36ea" />


# Data Preparation
## MinMaxScaler
MinMaxScaler is a data preprocessing technique used in machine learning to normalize data by scaling features to a specific, fixed range—typically 0 to 1. It is primarily used to ensure that all numerical features contribute equally to model training, preventing features with larger magnitudes from dominating those with smaller ranges.
It is essential for algorithms, such as multilayer perceptrons, that are sensitive to the magnitude of input values. It helps models converge faster during training, especially when using activation functions like Sigmoid or Tanh.

<img width="877" height="262" alt="image" src="https://github.com/user-attachments/assets/038c3769-6ccf-43f6-b8f3-9b00604a7137" />


# Model Building – LSTM
## What is long short-term memory (LSTM)?
A long short-term memory architecture (LSTM) is a special type of recurrent neural network (RNN) designed to learn and remember information over long sequences of data.
In time-series forecasting, LSTMs are still widely used for predicting future values in sequential data like stock prices and weather patterns. In healthcare, time series data can be analyzed to predict disease progression and treatment outcomes.

## Core Components of an LSTM Unit:
Cell State (Long-Term Memory):  A conveyor belt that runs through the entire chain, carrying information across time steps with minimal alteration, acting as the network's long-term memory.
Forget Gate:  Decides what information from the previous cell state should be thrown away (outputting a number between 0 and 1, where 0 means forget completely).
Input Gate:  Decides what new information from the current input and previous hidden state should be added to the cell state.
Output Gate: Filters the cell state to produce the final output (hidden state) for the current time step.

 <img width="674" height="424" alt="image" src="https://github.com/user-attachments/assets/337379e1-41a6-47f7-ba33-9b26c04417c2" />

 
<img width="849" height="499" alt="image" src="https://github.com/user-attachments/assets/bdd06dde-9c8e-4649-beeb-a5f55c181014" />


# Final Results
## Plotting loss function

 <img width="975" height="499" alt="image" src="https://github.com/user-attachments/assets/0113deca-c069-4106-a7c4-e58baa1b4c13" />

Here, Training and validation losses are almost close to each other. we can say,No issue of overfitting occurs here.

# Model evaluation
•	Test Loss: 0.0007680606795474887
•	MAPE: 0.02913992008677535
•	Accuracy: 0.9708600799132246
•	RMSE: 0.027713908218105154
•	MAE: 0.021671092021186687
•	R² Score: 0.9064418288078169


# Plotting Gold Price Forecasting with LSTM

<img width="981" height="528" alt="image" src="https://github.com/user-attachments/assets/1a58163a-3a18-41e2-8b9a-e8939a3e48d2" />


# Gold Price Future Forecasting with model
## forecasting gold price for next 6 months after 2023-01-01
## Model Forecasting-

<img width="1015" height="416" alt="image" src="https://github.com/user-attachments/assets/12103842-463d-4aa9-8824-04fd7c90fa05" />


## Actual Values- 

<img width="1048" height="560" alt="image" src="https://github.com/user-attachments/assets/07f7db52-f30c-44d9-aff5-0e346f3a9e1b" />


## As can seen, the model output is almost close to the actual changes in gold in 2023









