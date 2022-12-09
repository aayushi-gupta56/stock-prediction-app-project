--------------Stock-Market-Prediction-using-Sentiment-Analysis------------
Stock Market Prediction Web App based on Machine Learning and Sentiment Analysis of News headlines being web scraped from finviz. The front end of the Web App is based on Flask. 
The App forecasts stock prices of the next seven days for any given stock as input by the user. Predictions are made using three algorithms: ARIMA, LSTM, Linear Regression. The Web 
App combines the predicted prices of the next seven days with the sentiment analysis of news to give recommendation whether the price is going to rise or fall.


--------------File and Directory Structure----------------------
static - static files of flask app: css, images, js, etc.
templates - html files
app.py - main machine learning module


---------------Technologies Used---------------------
•	Flask
•	Tensorflow
•	Keras
•	Yahoo Finance
•	Alphavantage
•	Matplotlib
•	Nltk
•	StatsModels
•	Scikit-Learn
•	Python
•	CSS
•	HTML
•	Beautiful Soup


-----------------How to Install and Use------------------
1.	Clone the repo. 
2.	Go to command prompt, change directory to directory of repository and type pip install -r requirements.txt
3.	To run app, type in command prompt, python main.py
4.	Open your web browser and go to http://localhost:5000

