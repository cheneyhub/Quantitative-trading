#_*_conding:uft-8_*_
'''
Version: 0.0.1
Date: 2018-02-03
@Author:Cheney
'''

'''
A simple case to predict stock price rise/fall according to four Machine Learning models.
Analysis: 
I.   A classifier model, data from tushare stock data
II.  If stock's present price rising compared with previous day, mark the tag direction as 1, otherwise tag is -1
III. x_train/x_test is from percentage of lagged close price, y_train/y_test is from direction value   
Reference source: http://blog.csdn.net/u013547284/article/details/78443712

'''
import datetime
import numpy as np
import pandas as pd
import sklearn
import pandas_datareader as pdr
import tushare as ts

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC


def create_tag_data(symbol, start_date, end_date,lags=5):
    """
    It creates a pandas DataFrame that stores the percentage returns
    of the adjusted closing value of a stock obtained from Tushare.
    Lagged recent 5days
    :return:
    a DataFrame data with classified tag
    """

    #Obtain stock information from tushare
    tsdata = ts.get_hist_data(symbol, start_date,end_date)

    #Create new lagged DataFrame
    tslag = pd.DataFrame(index=tsdata.index)
    tslag['Today'] = tsdata['close']
    tslag['Volume'] = tsdata['volume']

    #Create shift lagged data
    for i in range(0, lags):
        tslag["Lag%s"%str(i+1)] = tsdata['close'].shift(i+1)

    # Create return DataFrame
    tsret = pd.DataFrame(index=tsdata.index)
    tsret['Volume'] = tslag['Volume']
    tsret['Today'] = tslag['Today'].pct_change()*100.0

    # Set a small number to prevent the values of return equal to zero
    for i, v in enumerate(tsret['Today']):
        if (abs(v) < 0.0001):
            tsret['Today'][i] = 0.0001

    # Create lagged percentage return columns
    for i in range(0, lags):
        tsret['Lag%s'%str(i+1)] = tslag['Lag%s'%str(i+1)].pct_change()*100.0

    # Create the direction columns(+1 or -1) indicates rise/fall
    tsret['Direction'] = np.sign(tsret['Today'])
    tsret = tsret[tsret.index >= start_date]
    tsret = tsret.dropna()

    print('Final data',tsret[:10])
    return tsret


if __name__ == "__main__":
    # Create a lagged data of S&P500 stock market index
    spret = create_tag_data(
        symbol="600036", start_date="2013-1-10",
        end_date="2016-12-31",lags=5
    )

    # Use the prior two days data and direction as train data
    x = spret[['Lag1','Lag2','Lag3','Lag4']]
    y = spret['Direction']

    start_test = "2015-1-1"
    x_train = x[x.index < start_test]
    x_test = x[x.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]


    # Create predict models
    print("Hit Rates/Confusion Matrices:\n")
    models = [('LR', LogisticRegression()),              
              ('LSVC', LinearSVC()),
              ('RSVM', SVC(C=10000.0, cache_size=300,class_weight=None,
                           coef0=0.0, gamma=0.001, kernel='rbf',
                           max_iter=-1, probability=False, random_state=None,
                           shrinking=True, tol=0.001, verbose=False)),
              ("RF", RandomForestClassifier(n_estimators=1000, criterion="gini",
                      max_depth=None, min_samples_split=2,min_samples_leaf=1,
                      max_features="auto", bootstrap=True, oob_score=False, n_jobs=1,
                      random_state=6, verbose=0))]

    # Iterate the models
    for model in models:
        model[1].fit(x_train, y_train)
        pred = model[1].predict(x_test)

        print("%s:\n%0.3f"%(model[0],model[1].score(x_test, y_test)))
        print("%s\n"%confusion_matrix(pred, y_test))

		
		
