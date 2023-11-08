import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as  mdates
from statsmodels.tsa.ar_model import AutoReg as AR
from matplotlib.dates import DateFormatter
import statsmodels.api as sn
from statsmodels.graphics.tsaplots import plot_acf
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

df_q1=pd.read_csv("daily_covid_cases.csv")
cases=df_q1['new_cases']
print("-----Q1-----")
#Q1 part a
fig, ax = plt.subplots()
ax.plot(df_q1['Date'],df_q1['new_cases'].values)
ax.set(xlabel="Month-Year", ylabel="New_cases",title="Lineplot--Q1a")
date_form = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation =45)
plt.show()

#Q1 part b
one_day_lag=cases.shift(1)
print("Pearson correlation (autocorrelation) coefficient :",cases.corr(one_day_lag))
print()

#Q1 part c
plt.scatter(cases, one_day_lag, s=5)
plt.xlabel("Given time series")
plt.ylabel("One day lagged time series")
plt.title("Q1 part c")
plt.show()

#Q1 part d
PCC=sn.tsa.acf(cases)
lag=[1,2,3,4,5,6]
pcc=PCC[1:7]
plt.plot(lag,pcc, marker='o')
for xitem,yitem in np.nditer([lag, pcc]):
        etiqueta = "{:.3f}".format(yitem)
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center")
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Q1 part d")
plt.show()

#Q1 part e
plot_acf(x=cases, lags=50)
plt.xlabel("Lag value")
plt.ylabel("Correlation coffecient value")
plt.title("Q1 part e")
plt.show()

def rms_error(x_pred,x_actual):
    return (((np.mean((x_pred-x_actual)**2))**.5)/(np.mean(x_actual)))*100

# def map_error(x_pred,x_actual):
#    return (np.mean(np.abs((x_actual-x_pred)/x_actual))/len(x_actual))*100

# Q2
print("-----Q2-----")
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35            # test size=35%
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# lag=5
ar_model=AR(train,lags=5).fit()
# finding the parametrs of autoregression
coef=ar_model.params
print("Q2 part a--> coefficients are :",coef)
history=train[len(train)-5:]
history=[history[k] for k in range(len(history))]
pred=list()
for t in range(len(test)):
    lag=[history[j] for j in range(len(history)-5,len(history))]
    yh=coef[0]
    for d in range(5):
        yh=yh+coef[d+1]*lag[5-d-1]
    obs=test[t]
    pred.append(yh)
    history.append(obs)

# Q2 part b, part 1
plt.scatter(test,pred)
plt.xlabel('Actual cases')
plt.ylabel('Predicted cases')
plt.title('Q2 part b, Part 1')
plt.show()

# Q2 part b, part 2
x=[i for i in range(len(test))]
plt.plot(x,test, label='Actual cases')
plt.plot(x,pred,label='Predicted cases')
plt.legend()
plt.title('Q2 part b, Part 2')
plt.show()

# Q2 part b, part 3
print("RMSE between actual and predicted test data: ",rms_error(pred,test))
print("MAPE between actual and predicted test data: ",mean_absolute_percentage_error(pred,test))

# Q3
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35            # test size=35%
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# function for creating Autoregression model, rmse, mape
def AutoRegression(train,test,i):
    ar_model=AR(train,lags=i).fit()
    # finding the parametrs of autoregression
    coef=ar_model.params
    history=train[len(train)-i:]
    history=[history[k] for k in range(len(history))]
    pred=list()
    for t in range(len(test)):
        lag=[history[j] for j in range(len(history)-i,len(history))]
        yh=coef[0]
        for d in range(i):
            yh=yh+coef[d+1]*lag[i-d-1]
        obs=test[t]
        pred.append(yh)
        history.append(obs)

    rms_e=rms_error(pred,test)
    map_e=mean_absolute_percentage_error(pred,test)
    return rms_e,map_e

print("-----Q3-----")
p=[1,5,10,15,25]            # p values
rms_list=[]
map_list=[]
for i in p:
    rmse,mape=AutoRegression(train,test,i)
    rms_list.append(rmse)
    map_list.append(mape)
# MAPE and RMSE values
print("MAPE values for p=1,5,10,15,25: ",map_list)
print("RMSE values for p=1,5,10,15,25: ",rms_list)
# plot for rmse
plt.bar(['1','5','10','15','25'],rms_list,width=.4)
plt.title("RMSE vs lag value")
plt.xlabel("lag value(p)")
plt.ylabel("RMSE")
plt.show()
# plot for mape
plt.bar(['1','5','10','15','25'],map_list,width=.4)
plt.title("MAPE vs lag value")
plt.xlabel("lag value(p)")
plt.ylabel("MAPE")
plt.show()

# Q4
print("-----Q4-----")
train_q4=series.iloc[:int(len(series)*0.65)]
train_q4=train_q4['new_cases']
i=0
corr = 1
# abs(AutoCorrelation) > 2/sqrt(T)
while corr > 2/(len(train_q4))**0.5:
    i += 1
    new_ = train_q4.shift(i)
    corr = train_q4.corr(new_)

print("Optimal value for lag is: ",i-1)

rms_q4,map_q4=AutoRegression(train,test,i-1)
print("RMSE Q4: ",rms_q4)
print("MAPE Q4: ",map_q4)