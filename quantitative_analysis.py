#_*_conding:uft-8_*_
'''
Version: 0.0.1
Date: 2018-01-28
@Author:Cheney
'''

'''
Machine learning for quantitative trading---stock price predict
Reference Source: 
http://mp.weixin.qq.com/s/yqSo8nF9fCDzuBREHsx1Cg

Before run the codes, you should make sure libraries like quandl, pytrends, pystan,fbprophet,
numpy, pandas and matplotlib have been installed. During install fbprophet library, it need to 
install "Microsoft Visual C++ Build Tools"
Build Tools Source: http://landinghub.visualstudio.com/visual-cpp-build-tools
'''

from quantitative_analysis_stocker import Stocker

#Get stock data, it will get Goldman stock date from 1999-05-04 to date
goldman= Stocker("GS")
print("Get Goldman stock data:\n",goldman)

#Draw Goldman history stock price plot
goldman.plot_stock()

#Prediction recent year stock price compared with observations
goldman.evaluate_prediction()

#Adjust changepoint Prior to select best model
goldman.changepoint_prior_analysis(changepoint_priors=[0.001,0.05,0.1,0.2])
goldman.changepoint_prior_validation(start_date="2016-01-04", end_date="2017-01-03",
                                    changepoint_priors=[0.001,0.05,0.1,0.2])

#According to evaluation result, assignment best parameter to model
goldman.changepoint_prior_scale = 0.1

#After optimizing the model parameter, use the model to predict stock price trendency
goldman.evaluate_prediction(nshares=1000)
goldman.predict_future(days=10)
goldman.predict_future(days=66)





