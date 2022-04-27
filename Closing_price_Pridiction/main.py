# -*- coding: utf-8 -*-
"""
THIS FILE CONTAINS THE FUNCTION FOR GET DATA CLEAN DATA
"""
#imports
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR 
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt 
from sklearn.model_selection import GridSearchCV
from constants import START_DATE,END_DATE,TICKER,COMPANY_NAMES
#use the style
plt.style.use('ggplot')


#functions

def get_data(Ticker):
    """To get a data from the yahoo finance
    of the company based on the ticker"""
    try :
        stock_data = data.DataReader(Ticker,
                                     'yahoo',
                                     START_DATE,
                                     END_DATE)
        
        return stock_data.reset_index()
    except RemoteDataError:
        print(f'No Data found for the {Ticker}')

def close_table(stock_data):
    """To create dependence of last price on the previous three days price """
    close1 = [i for i in stock_data["Close"]]
    close2 = [i for i in stock_data["Close"][1:]]
    close3 = [i for i in stock_data["Close"][2:]]
    predict = [i for i in stock_data["Close"][3:]]
    df = pd.DataFrame(list(zip(close1,close2,close3,predict)),columns =['Close1','Close2','Close3','Last_Close'])
    return df

def predict_Test(Ticker,dataFrame):
    df = close_table(get_data(Ticker))
    #reshaping of data
    x = df[["Close1","Close2","Close3"]].values
    y = df["Last_Close"].values
    print(df)
    # # Traing and test
    X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.50,random_state=42)

    
    #paramters for different kernel
    parameters_lin = [{'svr__C':[1,10,100,1000],'svr__epsilon':[0,0.01,0.1,0.5,1,2,4]}]
    parameters_rbf = [{'svr__C':[1,10,100,1000],  
                        'svr__gamma':np.logspace(-9, 3, 13)}] 
    parameters_poly =[{'svr__C': [1,10,100,1000],  
                        'svr__degree':[1,2,3,4,5],
                        'svr__gamma':['auto','scale']}] 
    
    #linear kernel scaling,hyperpaametr search 
    
    svr_lin =  make_pipeline(StandardScaler(),SVR(kernel='linear',cache_size=1000))
    gridsearch_lin = GridSearchCV(estimator=svr_lin,param_grid=parameters_lin,
                              scoring='r2',verbose=0, n_jobs=-1)
    
    #RBF kernel
    svr_rbf = make_pipeline(StandardScaler(), SVR(kernel= 'rbf',cache_size=1000))
    gridsearch_rbf = GridSearchCV(estimator=svr_rbf,param_grid=parameters_rbf,
                              scoring='r2',verbose=0, n_jobs=-1)
    #polynomial kernel
    svr_poly = make_pipeline(StandardScaler(), SVR(kernel='poly',cache_size=1000))
    gridsearch_poly = GridSearchCV(estimator=svr_poly,param_grid=parameters_poly,
                              scoring='r2',verbose=0, n_jobs=-1)
    #fiiting the data 
    gridsearch_lin.fit(X_train, y_train)
    gridsearch_rbf.fit(X_train, y_train)
    gridsearch_poly.fit(X_train, y_train)
    print(gridsearch_lin.transform())
    # predicted values of different kernel
    y_linear =gridsearch_lin.predict(X_test)  
    y_rbf = gridsearch_rbf.predict(X_test)
    y_poly = gridsearch_poly.predict(X_test)
     
    #  to evalute
    print("\n")
    print("After prediction!!")
    print(COMPANY_NAMES[TICKER.index(Ticker)])
    new_df = pd.DataFrame(columns=["y_test","Lin_pre","rbf_pre","poly_pre"])  
    new_df["y_test"] = y_test
    new_df['Lin_pre'] = y_linear
    new_df['rbf_pre'] = y_rbf
    new_df['poly_pre'] = y_poly
    print(new_df)
    print("----------------------------------------------------------")
    r2_linear = r2_score(y_test,y_linear)
    r2_rbf = r2_score(y_test,y_rbf)
    r2_poly = r2_score(y_test,y_poly)
    print("R-squared  linear: ",r2_linear)
    print("R-squared  RBF: ",r2_rbf)
    print("R-squared  poly: ",r2_poly)
    print("----------------------------------------------------------")
    mse_linear = sqrt(mean_squared_error(y_test, y_linear)) 
    mse_rbf = sqrt(mean_squared_error(y_test, y_rbf))
    mse_poly = sqrt(mean_squared_error(y_test, y_poly))
    print('Root mean square linear :', mse_linear)
    print('Root mean square RBF :', mse_rbf)
    print('Root mean square Poly :', mse_poly)
    print("----------------------------------------------------------")
    temp=[COMPANY_NAMES[TICKER.index(Ticker)],r2_linear,
                                  mse_linear,
                                  r2_rbf,
                                  mse_rbf,
                                  r2_poly,
                                  mse_poly,
                                  ]
    return temp

#predict for the all the nifty 50 Company
if __name__ == "__main__":
    #making the result 
        result = pd.DataFrame(columns = ['Company','linearR2','LinearMSE',
                                         'rbfR2','rbfMSE','polyR2','polyMSE'])
       
       
        j = 0  
        for i in TICKER:
           dict_temp = predict_Test(i,result)
           result.loc[j] = dict_temp
           j += 1
           break
        result.index = np.arange(1, len(result) + 1)
        print(result)
        result.plot(x = 'Company',y="R2",rot=60)
        result.to_csv("result.csv")
        



