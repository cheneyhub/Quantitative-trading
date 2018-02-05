# Quantitative-trading

This reposity will store quantitative trading strategy and cases used for stocks trading.

**************************************************************************
Quantitative analysis case
Before quantitative analysis, it needs to import stocker library.   
from quantitative_analysis_stocker import Stocker  

Get stock data from Stocker module
aapl_stock = Stocker("AAPL")   
aapl_stock.plot_stock()  #Draw Apple's Inc stock price plot


**************************************************************************
Caseaa_stock_predict_ml
The case use tushare stock data to predict stock price rise of fall according to different machine learning models.
The accuracy prediction of the case is around 50%, this case just offer the analysis process.

Codes of the case include two parts, one is for data fetch, the other one is main function to call machine learning models. 





