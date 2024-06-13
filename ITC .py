#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
import seaborn as sns


df = pd.read_excel("D:\\ITC.xlsx")
df.head(30) 


# In[24]:


itcdata13=df.dropna()
itcdata13.index=pd.to_datetime(itcdata13.Date)
itcdata13=itcdata13["Close"]['2013-01-01':'2013-12-2']
itcdata13.describe()


# In[53]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 13')
ax1.plot(itcdata13)


# In[56]:


rolLmean=itcdata13.rolling(12).mean()
rolLstd=itcdata13.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata13,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd, color='r',label='std')
plt.legend(loc='best')
plt.title("2013")
plt.show(block=False)


# In[14]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log13= np.log(itcdata13)
plt.plot(ts_log13)


# In[51]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff13= ts_log13-ts_log13.shift()
plt.plot(ts_log_diff13)

rolLmean= ts_log_diff13.rolling(12).mean()
rolLstd= ts_log_diff13.rolling(12).std()
org=plt.plot(ts_log_diff13,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std 2013')
plt.legend(loc='best')
plt.show(block=False)


# In[16]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff13,nlags=20)
lag_pacf=pacf(ts_log_diff13,nlags=20)


# In[17]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff13.dropna(),lags=40,ax=ax1)
ax2= fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff13.dropna(),lags=40,ax=ax2)


# In[28]:


itcdata14=df.dropna()
itcdata14.index=pd.to_datetime(itcdata14.Date)
itcdata14=itcdata14["Close"] ['2014-01-01':'2014-12-2']
itcdata14.describe()


# In[55]:


rolLmean=itcdata14.rolling(12).mean()
rolLstd=itcdata14.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata14,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd, color='r',label='std')
plt.legend(loc='best')
plt.title("2014")
plt.show(block=False)


# In[31]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log14= np.log(itcdata14)
plt.plot(ts_log14)


# In[57]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff14= ts_log14-ts_log14.shift()
plt.plot(ts_log_diff14)

rolLmean= ts_log_diff14.rolling(12).mean()
rolLstd= ts_log_diff14.rolling(12).std()
org=plt.plot(ts_log_diff14,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std 2014')
plt.legend(loc='best')
plt.show(block=False)


# In[34]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff14,nlags=20)
lag_pacf=pacf(ts_log_diff14,nlags=20)


# In[35]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff14.dropna(),lags=40,ax=ax1)
ax2= fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff14.dropna(),lags=40,ax=ax2)


# In[37]:


itcdata15= df.dropna()
itcdata15.index=pd.to_datetime(itcdata15.Date)
itcdata15=itcdata15["Close"] ['2015-01-01' : '2015-12-2']
itcdata15.describe()


# In[58]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 15')
plt.title("2015 ITC data")
ax1.plot(itcdata15)


# In[61]:


rolLmean= itcdata15.rolling(12).mean()
rolLstd= itcdata15.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata15,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2015")
plt.show(block=False)


# In[40]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log15= np.log(itcdata15)
plt.plot(ts_log15)


# In[62]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff15= ts_log15-ts_log15.shift()
plt.plot(ts_log_diff15)

rolLmean= ts_log_diff15.rolling(12).mean()
rolLstd= ts_log_diff15.rolling(12).std()
org=plt.plot(ts_log_diff15,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2015')
plt.legend(loc='best')
plt.show(block=False)


# In[65]:


ts_log_diff15.std()


# In[45]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff15,nlags=20)
lag_pacf=pacf(ts_log_diff15,nlags=20)


# In[50]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff15.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff15.dropna(),lags=40,ax=ax2)


# In[3]:


itcdata16= df.dropna()
itcdata16.index=pd.to_datetime(itcdata16.Date)
itcdata16=itcdata16["Close"] ['2016-01-01' : '2016-12-2']
itcdata16.describe()


# In[4]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 15')
plt.title("2016 ITC data")
ax1.plot(itcdata16)


# In[6]:


rolLmean= itcdata16.rolling(12).mean()
rolLstd= itcdata16.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata16,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2016")
plt.show(block=False)


# In[8]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log16= np.log(itcdata16)
plt.plot(ts_log16)


# In[30]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff16= ts_log16-ts_log16.shift()
plt.plot(ts_log_diff16)

rolLmean= ts_log_diff16.rolling(12).mean()
rolLstd= ts_log_diff16.rolling(12).std()
org=plt.plot(ts_log_diff16,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2016')
plt.legend(loc='best')
plt.show(block=False)


# In[11]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff16,nlags=20)
lag_pacf=pacf(ts_log_diff16,nlags=20)


# In[12]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff16.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff16.dropna(),lags=40,ax=ax2)


# In[14]:


itcdata17= df.dropna()
itcdata17.index=pd.to_datetime(itcdata17.Date)
itcdata17=itcdata17["Close"] ['2017-01-01' : '2017-12-2']
itcdata17.describe()


# In[15]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 17')
plt.title("2017 ITC data")
ax1.plot(itcdata17)


# In[17]:


rolLmean= itcdata17.rolling(12).mean()
rolLstd= itcdata17.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata17,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2017")
plt.show(block=False)


# In[18]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log17= np.log(itcdata17)
plt.plot(ts_log17)


# In[29]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff17= ts_log17-ts_log17.shift()
plt.plot(ts_log_diff17)

rolLmean= ts_log_diff17.rolling(12).mean()
rolLstd= ts_log_diff17.rolling(12).std()
org=plt.plot(ts_log_diff17,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2017')
plt.legend(loc='best')
plt.show(block=False)


# In[21]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff17,nlags=20)
lag_pacf=pacf(ts_log_diff17,nlags=20)


# In[22]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff17.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff17.dropna(),lags=40,ax=ax2)


# In[24]:


itcdata18= df.dropna()
itcdata18.index=pd.to_datetime(itcdata18.Date)
itcdata18=itcdata18["Close"] ['2018-01-01' : '2018-12-2']
itcdata18.describe()


# In[25]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 18')
plt.title("2018 ITC data")
ax1.plot(itcdata18)


# In[26]:


rolLmean= itcdata18.rolling(12).mean()
rolLstd= itcdata18.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)


org=plt.plot(itcdata18,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2018")
plt.show(block=False)


# In[27]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log18= np.log(itcdata18)
plt.plot(ts_log18)


# In[28]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff18= ts_log18-ts_log18.shift()
plt.plot(ts_log_diff18)

rolLmean= ts_log_diff18.rolling(12).mean()
rolLstd= ts_log_diff18.rolling(12).std()
org=plt.plot(ts_log_diff18,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2018')
plt.legend(loc='best')
plt.show(block=False)


# In[31]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff18,nlags=20)
lag_pacf=pacf(ts_log_diff18,nlags=20)


# In[32]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff18.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff18.dropna(),lags=40,ax=ax2)


# In[34]:


itcdata19= df.dropna()
itcdata19.index=pd.to_datetime(itcdata19.Date)
itcdata19=itcdata19["Close"] ['2019-01-01' : '2019-12-2']
itcdata19.describe()


# In[35]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 19')
plt.title("2019 ITC data")
ax1.plot(itcdata19)


# In[38]:


rolLmean= itcdata19.rolling(12).mean()
rolLstd= itcdata19.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata19,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2019")
plt.show(block=False)


# In[39]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log19= np.log(itcdata19)
plt.plot(ts_log19)


# In[40]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff19= ts_log19-ts_log19.shift()
plt.plot(ts_log_diff19)

rolLmean= ts_log_diff19.rolling(12).mean()
rolLstd= ts_log_diff19.rolling(12).std()
org=plt.plot(ts_log_diff19,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2019')
plt.legend(loc='best')
plt.show(block=False)


# In[41]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff19,nlags=20)
lag_pacf=pacf(ts_log_diff19,nlags=20)


# In[42]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff19.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff19.dropna(),lags=40,ax=ax2)


# In[44]:


itcdata20= df.dropna()
itcdata20.index=pd.to_datetime(itcdata20.Date)
itcdata20=itcdata20["Close"] ['2020-01-01' : '2020-12-2']
itcdata20.describe()


# In[45]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 20')
plt.title("2020 ITC data")
ax1.plot(itcdata20)


# In[46]:


rolLmean= itcdata20.rolling(12).mean()
rolLstd= itcdata20.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata20,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2020")
plt.show(block=False)


# In[47]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log20= np.log(itcdata20)
plt.plot(ts_log20)


# In[48]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff20= ts_log20-ts_log20.shift()
plt.plot(ts_log_diff20)

rolLmean= ts_log_diff20.rolling(12).mean()
rolLstd= ts_log_diff20.rolling(12).std()
org=plt.plot(ts_log_diff20,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2015')
plt.legend(loc='best')
plt.show(block=False)


# In[49]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff20,nlags=20)
lag_pacf=pacf(ts_log_diff20,nlags=20)


# In[50]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff20.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff20.dropna(),lags=40,ax=ax2)


# In[51]:


itcdata21= df.dropna()
itcdata21.index=pd.to_datetime(itcdata21.Date)
itcdata21=itcdata21["Close"] ['2021-01-01' : '2021-12-2']
itcdata21.describe()


# In[52]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 21')
plt.title("2021 ITC data")
ax1.plot(itcdata21)


# In[53]:


rolLmean= itcdata21.rolling(12).mean()
rolLstd= itcdata21.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata21,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2020")
plt.show(block=False)


# In[54]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log21= np.log(itcdata21)
plt.plot(ts_log21)


# In[56]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff21= ts_log21-ts_log21.shift()
plt.plot(ts_log_diff21)

rolLmean= ts_log_diff21.rolling(12).mean()
rolLstd= ts_log_diff21.rolling(12).std()
org=plt.plot(ts_log_diff21,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2021')
plt.legend(loc='best')
plt.show(block=False)


# In[57]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff21,nlags=20)
lag_pacf=pacf(ts_log_diff21,nlags=20)


# In[58]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff21.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff21.dropna(),lags=40,ax=ax2)


# In[59]:


itcdata22= df.dropna()
itcdata22.index=pd.to_datetime(itcdata22.Date)
itcdata22=itcdata22["Close"] ['2022-01-01' : '2022-12-2']
itcdata22.describe()


# In[60]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 22')
plt.title("2022 ITC data")
ax1.plot(itcdata22)


# In[61]:


rolLmean= itcdata22.rolling(12).mean()
rolLstd= itcdata22.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata22,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2022")
plt.show(block=False)


# In[63]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log22= np.log(itcdata22)
plt.plot(ts_log22)


# In[64]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff22= ts_log22-ts_log22.shift()
plt.plot(ts_log_diff22)

rolLmean= ts_log_diff22.rolling(12).mean()
rolLstd= ts_log_diff22.rolling(12).std()
org=plt.plot(ts_log_diff22,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2015')
plt.legend(loc='best')
plt.show(block=False)


# In[65]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff22,nlags=20)
lag_pacf=pacf(ts_log_diff22,nlags=20)


# In[66]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff22.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff22.dropna(),lags=40,ax=ax2)


# In[68]:


itcdata23= df.dropna()
itcdata23.index=pd.to_datetime(itcdata23.Date)
itcdata23=itcdata23["Close"] ['2023-01-01' : '2023-12-2']
itcdata23.describe()


# In[69]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.set_xlabel('TIME')
ax1.set_ylabel('ITC DATA 23')
plt.title("2023 ITC data")
ax1.plot(itcdata23)


# In[70]:


rolLmean= itcdata23.rolling(12).mean()
rolLstd= itcdata23.rolling(12).std()
plt.figure(figsize=(16,7))
figure=plt.figure(1)

org=plt.plot(itcdata23,color='b',label='org')
mean=plt.plot(rolLmean,color='g',label='mean')
std=plt.plot(rolLstd,color='r',label='std')
plt.legend(loc='best')
plt.title("Mean&std of 2023")
plt.show(block=False)


# In[71]:


plt.figure(figsize=(16,7))
fig = plt.figure(1)

ts_log23= np.log(itcdata23)
plt.plot(ts_log23)


# In[73]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff23= ts_log23-ts_log23.shift()
plt.plot(ts_log_diff23)

rolLmean= ts_log_diff23.rolling(12).mean()
rolLstd= ts_log_diff23.rolling(12).std()
org=plt.plot(ts_log_diff23,color='b',label='org')
mean=plt.plot(rolLmean, color='r',label='mean')
std= plt.plot(rolLstd,color='g',label='std')
plt.title('rolling mean & std of 2023')
plt.legend(loc='best')
plt.show(block=False)


# In[74]:


from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_diff20,nlags=20)
lag_pacf=pacf(ts_log_diff20,nlags=20)


# In[75]:


import statsmodels.api as sm
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(ts_log_diff23.dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(ts_log_diff23.dropna(),lags=40,ax=ax2)


# In[ ]:




