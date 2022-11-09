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

# data from Wegner expedition
# Wegener density #1
df_wegener = pd.read_excel('data/Wegener 1930/density_raw_Wegener.xlsx')
df_wegener['latitude'] = np.nan
df_wegener['longitude'] = np.nan

# adding latitude and longitude
tmp = pd.read_csv('data/Wegener 1930/snowdens_wegener_10-20-50_latlon.csv')
for i,dist in enumerate(tmp.distance_i):
    df_wegener.loc[df_wegener['distance to margin']==dist, 'latitude'] = tmp.loc[i,'lat']
    df_wegener.loc[df_wegener['distance to margin']==dist, 'longitude'] = tmp.loc[i,'lon']
    df_wegener['depth_mid'] = df_wegener['depth from'] + (df_wegener['depth to']-df_wegener['depth from'])/2
df_wegener.to_csv('data/Wegener 1930/density_raw_Wegener_with_lat_lon.csv')
   
# Sorge's density profile at Eismitte
df_dens_eismitte = pd.read_csv('data/Wegener 1930/digitize_firn_var_sources.csv', 
                               sep=';',
                               skiprows=3,
                               header=None).iloc[:,:2]
df_dens_eismitte.columns = ['depth','density']
df_dens_eismitte.depth = -df_dens_eismitte.depth
df_dens_eismitte.density = 1000* df_dens_eismitte.density

# 200km Randabstand

# other data
# not loading SUMup: only no additional cores compared to bav dataset
# df_sumup = pd.read_csv('../SUMup/SUMup 2023 beta/SUMup_greenland_density_2023.csv')
# msk = (df_sumup.longitude >-40.3) & (df_sumup.longitude <-39.3) \
#     & (df_sumup.latitude < 71.2) & (df_sumup.latitude > 70.8)\
#         & (df_sumup.elevation > 1800) 
# df_sumup = df_sumup.loc[msk, :]

# max_depth = df_sumup.groupby('profile')['stop_depth','midpoint'].max().mean(1)
# df_sumup = df_sumup.set_index('profile').loc[max_depth.loc[max_depth>5].index, : ]
# df_sumup.reference.unique()

# BAV dataset
df_bav_meta = pd.read_csv('../../Data/Cores/csv dataset/CoreList.csv', sep = ';')
df_bav_meta = df_bav_meta.loc[df_bav_meta.DensityAvailable==1]
df_bav_meta['longitude'] = df_bav_meta.Westing_decdeg
df_bav_meta['latitude'] = df_bav_meta.Northing_decdeg
df_bav_meta['elevation'] = df_bav_meta.Elevation_masl
msk = (df_bav_meta.longitude >-40.3) & (df_bav_meta.longitude <-39.3) \
    & (df_bav_meta.latitude < 71.2) & (df_bav_meta.latitude > 70.8)\
        & (df_bav_meta.MaxDepthm > 5) 
df_bav_meta = df_bav_meta.loc[msk, :]
df_bav_meta = df_bav_meta.set_index('CoreNo')
citation_bav = np.unique(df_bav_meta.Citation)

# %% Comparison of nearby density profiles at Eismitte
core_list = df_bav_meta.index.values

plt.figure(figsize=(4,6))
sym = 'o^dvs<>pP*ho^dvs<>pP*h'

df_morris = pd.DataFrame()
for i, core_id in enumerate(core_list[1:]):
    tmp = pd.read_csv('../../Data/Cores/csv dataset/cores/'+str(core_id)+'.csv', 
                           sep = ';',
                           names=['depth','density','type','ice_perc'],
                           na_values=-999)
    tmp['depth'] = tmp.depth/100
    tmp  = tmp[['depth','density']]
    tmp['Name'] = df_bav_meta.loc[core_id,'Name']
    df_morris =pd.concat((df_morris,
                          tmp))
df_morris_mean=df_morris.groupby('depth').density.mean()
df_morris_std=df_morris.groupby('depth').density.std()
plt.fill_betweenx(df_morris_mean.index, 
                  df_morris_mean-df_morris_std, 
                  df_morris_mean+df_morris_std,
                  color='lightgray')
plt.plot(df_morris_mean.values,
         df_morris_mean.index, 
         label='Morris and Wingham (2014)')

core_id = 348
tmp = pd.read_csv('../../Data/Cores/csv dataset/cores/'+str(core_id)+'.csv', 
                       sep = ';',
                       names=['depth','density','type','ice_perc'],
                       na_values=-999)
tmp['depth'] = tmp.depth/100
plt.plot(tmp.density, tmp.depth,
         #marker=sym[i], 
         label= 'Benson (1962)')
    
plt.plot(df_dens_eismitte.density, df_dens_eismitte.depth,
         linewidth=3, marker='o', label='Eismitte 1930')
# Eismitte source comparison
# The density profile from Eismitte was already in the firn density dataset
# as part of the Spencer et al. 2001 data
plt.legend()
plt.ylim(11, 0)
plt.xlabel('Density (kg m$^{-3}$)')
plt.ylabel('Depth (m)')
plt.savefig('plots/firn_density_eismitte.tif', dpi=300)


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
