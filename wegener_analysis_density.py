# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

#%% Firn density
df_sumup = pd.read_pickle('data/sumup_greenland_bav.pkl')
df_sumup=df_sumup.reset_index()

df_wegener = pd.read_excel('data/Wegener 1930/density_raw_Wegener.xlsx')
df_wegener['latitude'] = np.nan
df_wegener['longitude'] = np.nan

tmp = pd.read_csv('data/Wegener 1930/snowdens_wegener_10-20-50_latlon.csv')
for i,dist in enumerate(tmp.distance_i):
    df_wegener.loc[df_wegener['distance to margin']==dist, 'latitude'] = tmp.loc[i,'lat']
    df_wegener.loc[df_wegener['distance to margin']==dist, 'longitude'] = tmp.loc[i,'lon']
    df_wegener['depth_mid'] = df_wegener['depth from'] + (df_wegener['depth to']-df_wegener['depth from'])/2
df_bav=pd.read_csv('data/csv dataset/CoreList_no_sumup.csv',sep=";",encoding='ANSI')
df_bav = df_bav.loc[df_bav.DensityAvailable=='y']
citation_bav = np.unique(df_bav.Citation)

df_sumup[df_sumup==-9999.0]=np.nan
df_sumup = df_sumup.rename(columns=dict(lat='latitude',
                                        lon='longitude', 
                                        elev='elevation',
                                        top='depth_top',
                                        bot='depth_bot'))

#%%  Plotting location
df_sumup = df_sumup.loc[np.logical_and(df_sumup.longitude <-36.5,
                           np.logical_and(df_sumup.latitude < 73,
                                          df_sumup.latitude > 69)),:]
df_bav = df_bav.loc[np.logical_and(df_bav.longitude <-36.5,
                           np.logical_and(df_bav.latitude < 73,
                                          df_bav.latitude > 69)),:]

plt.figure()
# for i, txt in enumerate(df_sumup.name):
#     plt.annotate(#str(txt), #
#                   txt, 
#                   (df_sumup.longitude.iloc[i],
#                   df_sumup.latitude.iloc[i]))
plt.scatter(df_wegener.longitude,df_wegener.latitude,label='Wegener')
plt.scatter(df_bav.longitude,df_bav.latitude,label='other sources')
plt.scatter(df_sumup.longitude,df_sumup.latitude,label='sumup')
plt.xlabel('Longitude (deg E)')
plt.ylabel('Latitude (deg N)')
lg = plt.legend(loc='upper left',bbox_to_anchor=(0, 1.2),ncol=3,title='Density measurements')
lg.get_title().set_fontsize(15)

# %% plotting Wegener's density profiles

fig, ax = plt.subplots(6, int(len(df_wegener['distance to margin'].unique())/6+1), figsize=(18,9))
plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right = 0.95, top=0.95,bottom=0.05)
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]-1):
        ax[i,-1].yaxis.set_label_position("right")
        ax[i,-1].yaxis.tick_right()
        if (j>0):
            ax[i,j].yaxis.set_ticklabels([])
        if i<ax.shape[0]-1:
            ax[i,j].xaxis.set_ticklabels([])
ax=ax.flatten()

for i, dist in enumerate(df_wegener['distance to margin'].unique()):
    
    tmp = df_wegener.loc[dist==df_wegener['distance to margin'],:]
    for k in tmp.index:
        ax[i].plot(np.array([1, 1])*tmp.loc[k,'density'],
                 [tmp.loc[k,'depth from']/100,
                  tmp.loc[k,'depth to']/100],
                 color='tab:red')
    ax[i].set_ylim(2,0)
    ax[i].set_xlim(230,600)
#%% 
elev_bins = np.arange(1300,3000,200)
# plt.close('all')

fig, ax = plt.subplots(1,8,figsize=(15,10),sharey=True)
ax =ax.flatten()
fig.subplots_adjust(left=0.07, right = 0.98, 
                    top=0.9, bottom = 0.07, 
                    hspace=0.01, wspace=0.01)
import matplotlib
for i in range(len(elev_bins)-1):
    # wegener cores
    ind = (df_wegener.altitude>=elev_bins[i]) & (df_wegener.altitude<elev_bins[i+1])
    if np.sum(ind)>0:
        for k in df_wegener.loc[ind].index.values:
            ax[i].plot(np.array([1, 1])*df_wegener.loc[k,'density'],
                     [-df_wegener.loc[k,'depth from']/100,
                      -df_wegener.loc[k,'depth to']/100],
                     color='black',label='_nolegend_')
    
    sumup_year = pd.DatetimeIndex(df_sumup.date.values).year
    yr_bin = np.array([1950, 1980, 1990, 2000, 2010, 2020])
    cmap = matplotlib.cm.get_cmap('tab10')
    yr_col = np.array([cmap.colors[i] for i in range(len(yr_bin))])
    
    # sumup cores
    for j in range(len(yr_bin)):
        ind = np.logical_and(df_sumup.elevation>=elev_bins[i],
                                           df_sumup.elevation<elev_bins[i+1])
        if np.sum(ind)>0:
            for k in df_sumup.loc[ind].index.values:
                yr = df_sumup.loc[k,'date'].year
                ind_yr = np.logical_and(yr - yr_bin>0, yr - yr_bin<10)
                col = yr_col[ind_yr].tolist()[0]
                # print(yr, col)
                ax[i].plot(np.array([1, 1])*df_sumup.loc[k,'density']*1000,
                         [-df_sumup.loc[k,'depth_top'],
                          -df_sumup.loc[k,'depth_bot']],
                         color=col,
                         label='_nolegend_',zorder=50)           
    ax[i].set_title(str(elev_bins[i])+' - '+str(elev_bins[i+1])+' m a.s.l.')
    ax[i].set_xlim(300, 900)
    ax[i].set_ylim(-2, 0)
ax[i].plot(1, np.nan,color='black',label='Wegener 1930')
for j in range(len(yr_bin)):
    print(i,str(yr_bin[j])+' - '+str(yr_bin[j]+9))
    ax[i].plot(1, np.nan, color=yr_col[j].tolist(), label=str(yr_bin[j])+' - '+str(yr_bin[j]+9))
ax[4].set_xlabel('Density (kg m-3)')
ax[0].set_ylabel('Depth (m)')
handles, labels = ax[i].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',ncol=3)
