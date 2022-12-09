from flask import Flask, render_template, request, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.linear_model import LinearRegression
import statistics

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

#@app.after_request
#def add_header(response):
#    response.headers['Pragma'] = 'no-cache'
#    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#    response.headers['Expires'] = '0'
#    return response

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') 

@app.route('/results', methods=['POST'])
def results():
    def get_historical(quote):
        ticker = yf.Ticker(quote)
        df = ticker.history(period="15y")
        if(df.empty):
            ts = TimeSeries(key='8LGDYA5UN1L00K1I',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            data=data.head(4000).iloc[::-1] #Get last two years data
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
        print("Data Retrieval Successful..")
        return df

    def ARIMA_ALGO(df):
        df=df.reset_index()
        df=df.set_index("Code")
        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            forecast_set = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
                
            for t in range(7):
                model = ARIMA(history, order=(6, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                forecast_set.append(yhat)
                history.append(output)
            return predictions, forecast_set

        Quantity_date = df[['Close','Date']]
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'],axis =1)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(df['Date'], df['Close'])
        plt.savefig('static/Trends.png')
        plt.close(fig)
            
        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]
        
        predictions, forecast_set = arima_model(train, test)
            
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(test,label='Actual Price')
        plt.plot(predictions,label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ARIMA.png')
        plt.close(fig)
        arima_pred=round(forecast_set[0],2)
        error_arima = round(math.sqrt(mean_squared_error(test, predictions)),2)
        accuracy_arima = round((r2_score(test, predictions)*100),2)
        mean = statistics.mean(forecast_set)

        print("ARIMA Model Retrieval Successful..")
        return arima_pred, error_arima, accuracy_arima, forecast_set, mean
    
    def LSTM_ALGO(df):
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        training_set=df.iloc[:,3:4].values #Taking Adj Close values for all rows
    
        sc=MinMaxScaler(feature_range=(0,1))
        training_set_scaled=sc.fit_transform(training_set) #First fit to data and then transform for training data
    
        X_train=[]
        y_train=[]
    
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        
        X_train=np.array(X_train)
        y_train=np.array(y_train)
    
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        
        regressor=Sequential()
        
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.1))
        
        #Add 2nd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 3rd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        #Add o/p layer
        regressor.add(Dense(units=1))
        
        #Compile
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        #Training
        regressor.fit(X_train,y_train,epochs=10,batch_size=32)
    
        #For lstm, batch_size=power of 2
        #Testing
        real_stock_price=dataset_test.iloc[:,3:4].values
        dataset_total = df.iloc[:,3:4]
        testing_set = dataset_total[ len(dataset_total)-len(dataset_test)-7: ].values
        testing_set=testing_set.reshape(-1,1)
        testing_set=sc.transform(testing_set)
        
        X_test=[]
        y_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            y_test.append(testing_set[i, 0])
    
        X_test=np.array(X_test)
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        predicted_stock_price=regressor.predict(X_test)
    
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(real_stock_price,label='Actual Price')  
        plt.plot(predicted_stock_price,label='Predicted Price')
          
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        error_lstm = round(math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price)),2)
        
        forecasted_stock_price = regressor.predict(X_forecast)
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
        lstm_pred = round(forecasted_stock_price[0,0],2)
        accuracy_lstm = round(r2_score(real_stock_price, predicted_stock_price)*100, 2)

        print("LSTM Model Retrieval Successful..")
        return lstm_pred, error_lstm, accuracy_lstm

    def LIN_REG_ALGO(df):
        forecast_out = int(7)
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        df_new=df[['Close','Close after n days']]

        y = np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
            
            #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
            
            #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
            
            # Feature Scaling===Normalization
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
            
            #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
            
            #Testing
        y_test_pred=clf.predict(X_test)
        
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(y_test,label='Actual Price' )
        plt.plot(y_test_pred,label='Predicted Price')
            
        plt.legend(loc=4)
        plt.savefig('static/LR.png')
        plt.close(fig)
            
        error_lr = round(math.sqrt(mean_squared_error(y_test, y_test_pred)), 2)
            
            
            #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        mean=forecast_set.mean()
        lr_pred=round(forecast_set[0,0], 2)
        accuracy_lr = round(r2_score(y_test, y_test_pred)*100, 2)

        print("LR Model Retrieval Successful..")
        return lr_pred, error_lr, accuracy_lr

    def sentimentAnalysis(ticker):
        vader = SentimentIntensityAnalyzer()
        finviz_url = 'https://finviz.com/quote.ashx?t='
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent':'my-app'})
        response = urlopen(req)
        html = BeautifulSoup(response, features="lxml")
        news_table = html.find(id='news-table')
        news_rows = news_table.findAll('tr')
        news_list=[]
        pos = 0
        neg = 0
        neut = 0
        global_polarity = 0.0 
        
        for index, row in enumerate(news_rows):
            title = row.a.get_text()
            news_list.append(title)
            compound = vader.polarity_scores(title)["compound"]
            global_polarity = global_polarity + compound
            if(compound>0):
                pos = pos + 1
            elif(compound<0):
                neg = neg + 1
            else:
                neut = neut + 1
        labels=['Positive','Negative','Neutral']
        sizes = [pos,neg,neut]
        explode = (0, 0, 0)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  
        plt.tight_layout()
        plt.savefig('static/SA.png')
        plt.close(fig)

        
        if global_polarity >= 0:
            news_pol = "OVERALL POSITIVE"
        else:
            news_pol = "OVERALL NEGATIVE"

        print("Sentiment Analysis Retrieval Successful..")
        return global_polarity, news_list, pos, neg, neut, news_pol

    
    def recommendation(pos, neg, neut, quote_data, mean):
        today_stock = quote_data.iloc[-1:]
        if today_stock.iloc[-1]['Close'] < mean:
            if neg>pos and neg>neut:
                idea="FALL"
                decision="SELL"
            else:
                idea="RISE"
                decision="BUY"
        else:
            idea= "FALL"
            decision= "SELL"

        print("Recommendation Retrieval Successful..")
        return idea, decision

    quote = request.form['ticker']
    quote_data = get_historical(quote)
    today_stock = quote_data.iloc[-1:]
    today_stock = today_stock.round(2)
    quote_data = quote_data.dropna()
    quote_data["Code"] = quote
    arima_pred, error_arima, accuracy_arima, forecast_set, mean = ARIMA_ALGO(quote_data)
    lstm_pred, error_lstm, accuracy_lstm = LSTM_ALGO(quote_data)
    lr_pred, error_lr, accuracy_lr = LIN_REG_ALGO(quote_data)
    global_polarity, news_list, pos, neg, neut, news_pol = sentimentAnalysis(quote)
    idea, decision = recommendation(pos, neg, neut, quote_data, mean)
    total_items = len(news_list)

    return render_template('results.html',quote=quote,arima_pred=arima_pred,lstm_pred=lstm_pred,
                            lr_pred=lr_pred, open_s=today_stock['Open'].values[0],
                            close_s=today_stock['Close'].values[0], 
                            high_s=today_stock['High'].values[0], low_s=today_stock['Low'].values[0],
                            vol=today_stock['Volume'].values[0], error_lstm=error_lstm,error_arima=error_arima, error_lr=error_lr,
                            accuracy_arima=accuracy_arima, accuracy_lstm=accuracy_lstm, accuracy_lr=accuracy_lr, 
                            global_polarity = global_polarity, news_list = news_list, pos = pos, neg = neg, neut = neut, forecast_set=forecast_set, mean=mean, 
                            total_items= total_items, news_pol= news_pol, idea=idea, decision=decision)

    

if __name__ == '__main__':
    app.run()