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
from scipy.interpolate import interp1d

def interp_pandas(s, kind="quadratic"):
    # A mask indicating where `s` is not null
    m = s.notna().values
    s_save = s.copy()
    # Construct an interpolator from the non-null values
    # NB 'kind' instead of 'method'!
    kw = dict(kind=kind, fill_value="extrapolate")
    f = interp1d(s[m].index, s.loc[m].values.reshape(1, -1)[0], **kw)

    # Apply this to the indices of the nulls; reconstruct a series
    s[~m] = f(s[~m].index)[0]

    plt.figure()
    s.plot(marker="o", linestyle="none")
    s_save.plot(marker="o", linestyle="none")
    # plt.xlim(0, 60)
    return s

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
df_temp.to_csv('data/10m_temp_data_subset.csv')


#  Displaying temperature measurements
import matplotlib.text as mtext
df_200 = pd.read_csv('data/Wegener 1930/temperature/200mRandabst_firtemperature_wegener.csv', sep=';')
df_200.depth = df_200.depth/100

df1 = df_200.append({"Firntemp": np.nan, "depth": 10}, ignore_index=True)
df1 = interp_pandas(df1.set_index("depth"), kind="quadratic")
# linear: -23.9
# cubic: -25.6
# quadratic: -23.06
df_eism = pd.read_csv('data/Wegener 1930/temperature/Eismitte_digitize_firntemperatures_wegener.csv', sep=';')
df_eism['date'] = ['1930-'+str(m)+'-15' for m in df_eism.month.astype(str)]
df_eism = df_eism.set_index('date').drop(columns='month').T
df_eism.index=np.arange(0,16)
df_eism = -df_eism

plt.close('all')

fig,ax=plt.subplots(2,2,figsize=(8,7))
ax =ax.flatten()
ax[3].set_axis_off()
plt.subplots_adjust(bottom=0.08, left=0.12, top=0.97, wspace=0.08)

# =======================  temperature profile =================================
ax[0].invert_yaxis()
ax[0].plot(np.nan,np.nan, color='w', linestyle='None', label='RA200:')
df_200.plot(ax=ax[0], y='depth', x='Firntemp',  
            color = 'tab:blue',
            label='1930-07-24')
ax[0].plot(-23.9, 10, color='tab:blue',
           marker='d', label='_nolegend_')

ax[0].plot(np.nan,np.nan, color='w', linestyle='None', label='Eismitte:')
cmap = cm.get_cmap('autumn')
tmp = df_temp.loc[df_temp.site=='Eismitte', :]
plot_lines = []
for i, date in enumerate(df_eism.columns[:7]):
    df_eism[date].reset_index().plot(ax=ax[0], x=date, y='index', 
                                     color=cmap(i/7), label=date)
    if len(tmp.loc[tmp.date == date,'temperatureObserved'].values) == 1:
        ax[0].plot(tmp.loc[tmp.date == date,'temperatureObserved'].values,
               10, color=cmap(i/7), marker='d', label='_nolegend_')
ax[0].plot(np.nan,np.nan, color='k', marker='d', linestyle='None', label='interp. at 10 m')

    
ax[0].legend(loc='lower right', fontsize=9)
ax[0].grid()
ax[0].set_ylim(16, -0.3)
ax[0].set_xlabel('Firn temperature ($^o$C)', size=12)
ax[0].set_ylabel('Depth (m)', size=12)

# =======================  T10m =================================
pos = ax[2].get_position()
ax[2].set_position([pos.x0, pos.y0, 0.4, 0.5])
sym = 'o^dvs<>pP*ho^dvs<>pP*h'
cmap = cm.get_cmap('Spectral_r')
T10m_list = []
for i, year in enumerate(range(1930,2020,10)):
    msk = (df_temp.year >= year) & (df_temp.year < year+10)
    tmp = df_temp.loc[msk, :]
    if len(tmp)<2:
        continue
    if (tmp.elevation.max() - tmp.elevation.min())<500:
        continue
    if i == 0:
        zorder=1000
    for longitude in tmp.longitude.unique():
        if len(T10m_list)==0:
            T10m_list = tmp.loc[tmp.longitude == longitude, 
                                ['site','latitude','longitude', 'reference_short']].iloc[:1,:]
        else:
            T10m_list = pd.concat((T10m_list, tmp.loc[tmp.longitude == longitude, 
                                ['site','latitude','longitude', 'reference_short']].iloc[:1,:]))
        
    for k, ref in enumerate(tmp.reference_short.unique()):

                
        msk = (tmp.reference_short == ref)
        tmp1 = tmp.loc[msk, :]
        if i ==0:
            ax[2].plot( tmp1.temperatureObserved, tmp1.elevation,
                     marker=sym[k],color=cmap(i/8),markersize=8, zorder=1000, linestyle ='None',label=ref)
        else:
            ax[2].plot(tmp1.temperatureObserved, tmp1.elevation, 
                     marker=sym[2],color=cmap(i/8), linestyle ='None',
                     label='_nolegend_')

    
    d = np.polyfit(tmp.elevation, tmp.temperatureObserved, 1)
    print(np.round(d[0]*100,2),'for '+str(year)+'-'+str(year+9))
    f = np.poly1d(d)
    if i == 0:
        ax[2].plot(f([1800, 3300]),[1800, 3300], color=cmap(i/8), zorder=1000,
                   label='fit '+str(year)+'-'+str(year+9) )
    else:
        ax[2].plot(np.nan, np.nan, marker=sym[2], color=cmap(i/8),
                   linestyle ='None',label='Obs. '+str(year)+'-'+str(year+9) )
        ax[2].plot(f([1800, 3300]),[1800, 3300], color=cmap(i/8), 
                   zorder=0, label='fit '+str(year)+'-'+str(year+9) )
T10m_list.to_csv('T10m_sites_coords.csv', index=None)
       
ax[2].grid()
ax[2].legend(ncol=1, loc='lower left', bbox_to_anchor=(1.01 , -0.2), fontsize=9)
ax[2].set_xlabel('10 m firn temperature ($^o$C)', size=12)
ax[2].set_ylabel('Elevation (m a.s.l.)', size=12)

# =======================  density =================================
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

df_eismitte = pd.read_excel('data/Wegener 1930/schnee-firndichte_wegener.xlsx').iloc[1:, :]
df_eismitte.columns =['Nr.', 'date', 'depth_from', 'depth_to',  'note', 'density', 'note_bav']
df_eismitte[['depth_from', 'depth_to']] = df_eismitte[['depth_from', 'depth_to']]/100
df_eismitte.density = 1000* df_eismitte.density

for i in df_eismitte.index:
    if df_eismitte.loc[i, 'depth_from'] <0:
        print('On', df_eismitte.loc[i, 'date'],'the surface had increased by', - df_eismitte.loc[i, 'depth_from'])
        df_eismitte.loc[i:, ['depth_from','depth_to']] = df_eismitte.loc[i:, ['depth_from','depth_to']] - df_eismitte.loc[i, 'depth_from']

# not loading SUMup: only no additional cores compared to bav dataset
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

# % Comparison of nearby density profiles at Eismitte
core_list = df_bav_meta.index.values
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
ax[1].fill_betweenx(df_morris_mean.index, 
                  df_morris_mean-df_morris_std, 
                  df_morris_mean+df_morris_std,
                  color='lightsalmon')
ax[1].plot(df_morris_mean.values,
         df_morris_mean.index, 
         color='red',
         linewidth=1,
         label='T35 2004-2010\n(Morris and Wingham, 2014)')

core_id = 348
tmp = pd.read_csv('../../Data/Cores/csv dataset/cores/'+str(core_id)+'.csv', 
                       sep = ';',
                       names=['depth','density','type','ice_perc'],
                       na_values=-999)
tmp['depth'] = tmp.depth/100
ax[1].plot(tmp.density, tmp.depth,
         color='tab:blue',
         linewidth=1.5,
         label= 'T35 1955 (Benson, 1962)')

for i, month in enumerate([9, 10, 11, 12, 1, 2, 3, 4]):
    tmp = df_eismitte.set_index('date')
    tmp = tmp.loc[tmp.index.month==month, :]
    if tmp.shape[0] == 0:
        continue
    for row in tmp[['depth_from','depth_to','density']].iterrows():
        row = row[1]
        ax[1].plot([row['density'], row['density']], 
                 row[['depth_from','depth_to']], 
                 color='k',
                 linewidth=2, label='_noledgend_')

ax[1].plot(df_dens_eismitte.density, df_dens_eismitte.depth*np.nan,
         linewidth=2, color='k', label='Eismitte 1930-31 (Sorge, 1935)')
ax[1].grid()
# Eismitte source comparison
# The density profile from Eismitte was already in the firn density dataset
# as part of the Spencer et al. 2001 data
ax[1].legend(loc='lower left', fontsize=9)
ax[1].set_ylim(16, 0)
ax[1].set_xlabel('Density (kg m$^{-3}$)',size=12)
ax[1].set_ylabel('Depth (m)',size=12)
l, b, w, h = ax[0].get_position().bounds
ax[0].set_position([l-0.05,        b-0.15, w, h+0.15])
l1, _, _, _ = ax[1].get_position().bounds
ax[1].set_position([l1,  b-0.15, w*1.2,     h+0.15])
l, b, w, h = ax[2].get_position().bounds
ax[2].set_position([l, b, w*1.65, h*0.45])

ax[0].set_ylim(16, 0)
ax[1].set_ylim(16, 0)
plt.annotate('A', (0.025,0.94), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.annotate('B', (0.08,0.3), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.annotate('C', (0.475,0.94), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.savefig('plots/firn_density_temperature.tif', dpi=300)


