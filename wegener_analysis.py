# -*- coding: utf-8 -*-
"""
@author:  bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# loading the T10m dataset
df_temp = pd.read_csv('C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Data/Firn temperature/output/10m_temperature_dataset_monthly.csv')
msk = (df_temp.longitude <-36.5) & (df_temp.latitude < 73) & (df_temp.latitude > 69) & (df_temp.elevation > 1800) & (df_temp.date < '2021')
df_temp = df_temp.loc[msk, :]
df_temp[df_temp == -9999] = np.nan
df_temp['date'] = pd.to_datetime(df_temp.date)
year = df_temp.latitude.values * np.nan
for i, x in enumerate(df_temp.date):
    try:
        year[i] = x.year
    except:
        year[i] = x
ind = np.logical_or(df_temp.site == 'Eismitte',
                    df_temp.site == '200 km Randabstand')
df_temp['year'] = year
df_temp.to_csv('out/10m_temp_data_subset.csv')

# %% Displaying temperature measurements
import matplotlib.text as mtext
df_200 = pd.read_csv('data/Wegener 1930/temperature/200mRandabst_firtemperature_wegener.csv', sep=';')
df_200.depth = df_200.depth/100
df_eism = pd.read_csv('data/Wegener 1930/temperature/Eismitte_digitize_firntemperatures_wegener.csv', sep=';')
df_eism['date'] = ['1930-'+str(m)+'-15' for m in df_eism.month.astype(str)]
df_eism = df_eism.set_index('date').drop(columns='month').T
df_eism.index=np.arange(0,16)
df_eism = -df_eism

plt.close('all')

fig,ax=plt.subplots(1,2,figsize=(12,5))
plt.subplots_adjust(left=0.07, top=0.97, wspace=0.08)
ax[0].invert_yaxis()
ax[0].plot(np.nan,np.nan, color='w', linestyle='None', label='200km Randabstand:')
df_200.plot(ax=ax[0], y='depth', x='Firntemp',  
            color = 'tab:blue',
            label='1930-07-24')
ax[0].plot(df_temp.loc[df_temp.site=='200 km Randabstand', 'temperatureObserved'].values[0],
           10, color='tab:blue',
           marker='d', label='_nolegend_')

ax[0].plot(np.nan,np.nan, color='w', linestyle='None', label='Eismitte:')
cmap = cm.get_cmap('magma')
tmp = df_temp.loc[df_temp.site=='Eismitte', :]
plot_lines = []
for i, date in enumerate(df_eism.columns):
    df_eism[date].reset_index().plot(ax=ax[0], x=date, y='index', 
                                     color=cmap(i/12), label=date)
    if len(tmp.loc[tmp.date == date,'temperatureObserved'].values) == 1:
        ax[0].plot(tmp.loc[tmp.date == date,'temperatureObserved'].values,
               10, color=cmap(i/12), marker='d', label='_nolegend_')
ax[0].plot(np.nan,np.nan, color='k', marker='d', linestyle='None', label='interp. at 10 m')

    
ax[0].legend(loc='lower right')
ax[0].grid()
ax[0].set_ylim(16, -0.3)
ax[0].set_xlabel('Firn temperature ($^o$C)', size=12)
ax[0].set_ylabel('Depth (m)', size=12)

# 10 m firn temperature anaysis
pos = ax[1].get_position()
ax[1].set_position([pos.x0, pos.y0, 0.4, 0.5])
sym = 'o^dvs<>pP*ho^dvs<>pP*h'
cmap = cm.get_cmap('Spectral')
for i, year in enumerate(range(1930,2020,10)):
    msk = (df_temp.year >= year) & (df_temp.year < year+10)
    tmp = df_temp.loc[msk, :]
    if len(tmp)<2:
        continue
    if (tmp.elevation.max() - tmp.elevation.min())<500:
        continue
    for k, ref in enumerate(tmp.reference_short.unique()):
        msk = (tmp.reference_short == ref)
        tmp1 = tmp.loc[msk, :]
        if i ==0:
            ax[1].plot(tmp1.elevation, tmp1.temperatureObserved,
                     marker=sym[k],color=cmap(i/8),markersize=12, linestyle ='None',label=ref)
        else:
            ax[1].plot(tmp1.elevation, tmp1.temperatureObserved,
                     marker=sym[k],color=cmap(i/8), linestyle ='None',label=ref)
    
    d = np.polyfit(tmp.elevation, tmp.temperatureObserved, 1)
    f = np.poly1d(d)
    ax[1].plot([1800, 3300],f([1800, 3300]),color=cmap(i/8), label='fit '+str(year)+'-'+str(year+9) )
ax[1].grid()
ax[1].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5,1.01), fontsize=9)
ax[1].set_ylabel('10 m firn temperature ($^o$C)', size=12)
ax[1].set_xlabel('Elevation (m a.s.l.)', size=12)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
plt.annotate('A', (0.03,0.94), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.annotate('B', (0.51,0.65), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.savefig('plots/firn_temperature.tif', dpi=300)
#%% Firn density
df_sumup = pd.read_pickle('C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/SUMup/SUMup 2023 beta/sumup_greenland_bav.pkl')
df_sumup=df_sumup.reset_index()

df_wegener = pd.read_excel('Wegener/density_raw_Wegener.xlsx')
df_wegener['latitude'] = np.nan
df_wegener['longitude'] = np.nan

tmp = pd.read_csv('Wegener/snowdens_wegener_10-20-50_latlon.csv')
for i,dist in enumerate(tmp.distance_i):
    df_wegener.loc[df_wegener['distance to margin']==dist, 'latitude'] = tmp.loc[i,'lat']
    df_wegener.loc[df_wegener['distance to margin']==dist, 'longitude'] = tmp.loc[i,'lon']
    df_wegener['depth_mid'] = df_wegener['depth from'] + (df_wegener['depth to']-df_wegener['depth from'])/2
df_bav=pd.read_csv('data/csv dataset/CoreList_no_sumup.csv',sep=";",encoding='ANSI')
df_bav = df_bav.loc[df_bav.DensityAvailable=='y']
citation_bav = np.unique(df_bav.Citation)

    
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
#%% 
elev_bins = np.arange(1300,3000,200)

fig, ax = plt.subplots(1,8,figsize=(20,12),sharey=True)
ax =ax.flatten()
fig.subplots_adjust(left=0.07, right = 0.98, 
                    top=0.9, bottom = 0.07, 
                    hspace=0.01, wspace=0.01)
import matplotlib
for i in range(len(elev_bins)-1):
    # wegener cores
    ind = np.logical_and(df_wegener.altitude>=elev_bins[i],
                                           df_wegener.altitude<elev_bins[i+1])
    if np.sum(ind)>0:
        for k in df_wegener.loc[ind].index.values:
            ax[i].plot(np.array([1, 1])*df_wegener.loc[k,'density'],
                     [-df_wegener.loc[k,'depth from']/100,
                      -df_wegener.loc[k,'depth to']/100],
                     color='black',label='_nolegend_')
    
    sumup_year = pd.DatetimeIndex(df_sumup.date.values).year
    yr_bin = np.array([1950, 1980, 1990, 2000, 2010])
    cmap = matplotlib.cm.get_cmap('tab10')
    yr_col = np.array([cmap.colors[i] for i in range(len(yr_bin))])
    # sumup cores
    for j in range(len(yr_bin)):
        ind = np.logical_and(df_sumup.elevation>=elev_bins[i],
                                           df_sumup.elevation<elev_bins[i+1])
        if np.sum(ind)>0:
            for k in df_sumup.loc[ind].index.values:
                yr = df_sumup.loc[k,'date'].year
                ind_yr = np.logical_and(yr - yr_bin>0,yr - yr_bin<10)
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
