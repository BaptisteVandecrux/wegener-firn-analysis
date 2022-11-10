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


# %% Displaying temperature measurements
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

fig,ax=plt.subplots(1,2,figsize=(12,5))
plt.subplots_adjust(left=0.07, top=0.97, wspace=0.08)
ax[0].invert_yaxis()
ax[0].plot(np.nan,np.nan, color='w', linestyle='None', label='200km Randabstand:')
df_200.plot(ax=ax[0], y='depth', x='Firntemp',  
            color = 'tab:blue',
            label='1930-07-24')
ax[0].plot(-23.9, 10, color='tab:blue',
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
cmap = cm.get_cmap('Spectral_r')
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
            ax[1].plot( tmp1.temperatureObserved, tmp1.elevation,
                     marker=sym[k],color=cmap(i/8),markersize=12, linestyle ='None',label=ref)
        else:
            ax[1].plot(tmp1.temperatureObserved, tmp1.elevation, 
                     marker=sym[k],color=cmap(i/8), linestyle ='None',label=ref)
    
    d = np.polyfit(tmp.elevation, tmp.temperatureObserved, 1)
    print(np.round(d[0]*100,2),'for '+str(year)+'-'+str(year+9))
    f = np.poly1d(d)
    ax[1].plot(f([1800, 3300]),[1800, 3300],color=cmap(i/8), label='fit '+str(year)+'-'+str(year+9) )
ax[1].grid()
ax[1].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5,1.01), fontsize=9)
ax[1].set_xlabel('10 m firn temperature ($^o$C)', size=12)
ax[1].set_ylabel('Elevation (m a.s.l.)', size=12)
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
plt.annotate('A', (0.03,0.94), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.annotate('B', (0.51,0.65), xytext=None, xycoords='figure fraction',
               size=14, weight='bold')
plt.savefig('plots/firn_temperature.tif', dpi=300)
