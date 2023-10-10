import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import datetime
import warnings

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 13






def get_moments(rets,FREQ=1,doStyle=True):
    moments = pd.concat([rets.mean() * FREQ, rets.std() * np.sqrt(FREQ), rets.skew(), rets.kurtosis()],axis=1,keys=['mean','vol','skewness','kurtosis'])

    if doStyle:
        moments = moments.style.format({'mean':'{:.2%}','vol':'{:.2%}','skewness':'{:.2f}','kurtosis':'{:.2f}'})
    
    return moments




def bivariate_risk(rets,keyX):
    
    birisk = pd.DataFrame(dtype=float, columns=['corr','cov','beta'], index=rets.columns)
    birisk['corr'] = rets.corr()[keyX]
    birisk['cov'] = rets.cov()[keyX]
    for sec in rets.columns:
        birisk.loc[sec,'beta'] = LinearRegression().fit(rets[[keyX]],rets[[sec]]).coef_[0]

    birisk.columns = [f'{keyX} {col}' for col in birisk.columns]
    birisk.style.format({birisk.columns[0]:'{:.2%}',birisk.columns[1]:'{:.4%}',birisk.columns[2]:'{:.4f}'})
    
    return birisk




def mdd_timeseries(rets):
    cum_rets = (1 + rets).cumprod()
    rolling_max = cum_rets.cummax()
    drawdown = (cum_rets - rolling_max) / rolling_max
    return drawdown







def plot_normal_histogram(data,bins=10):
    fig, ax, = plt.subplots()
    mu, std = norm.fit(data)
    data.plot.hist(ax=ax,density=True,bins=bins);
    xmin, xmax = plt.xlim();
    x = np.linspace(xmin,xmax,100)
    npdf = norm.pdf(x,mu,std)
    plt.plot(x,npdf,'k',linewidth=3);
    plt.ylim(0,max(npdf))
    plt.legend(['sample','normal'])
    return fig;







def outlier_normal(data,doStyle=True):
    z = (data - data.mean())/data.std()
    phi = z.apply(norm.cdf)

    prob_outlier = pd.DataFrame(columns=['normal prob min','normal prob max'],index=data.columns,dtype=float)
    for i in phi.columns:
        prob_outlier.loc[i] = [phi[i].min(),1-phi[i].max()]

    prob_outlier.T

    temp = pd.concat([z.min(),z.max()],axis=1)
    temp.columns=['z min','z max']
    
    outlier_norm = pd.concat([temp,prob_outlier],axis=1)
    
    if doStyle:
        outlier_norm = outlier_norm.style.format({'z min':'{:.2f}','z max':'{:.2f}','normal prob min':'{:.2e}','normal prob max':'{:.2e}'})
        
    return outlier_norm








def test_coherence(rets,keys,quantile=.05):

    comp = pd.concat([rets[keys],rets[keys].sum(axis=1)],axis=1).rename(columns={0:'Portfolio'})

    mdd = mdd_timeseries(comp)
    

    tabcomp = pd.concat([comp.std(), comp.var(), comp.quantile(quantile), mdd.min()],axis=1,keys=['std','variance',f'quantile {quantile}','MDD'])
    tabcomp.loc['sum of parts'] = tabcomp.iloc[0:-1,:].sum()

    return tabcomp