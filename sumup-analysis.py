"""
Created on %(date)s

@author: bav

SUMup dataset processing

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# %%  Load the data.
### There are two pickles, each containing a pandas dataframe with the core data.
dfg = pd.read_pickle('data/sumup_greenland.pkl')
dfg[dfg==-9999]=np.nan
sumup_citations = pd.read_csv('data/sumup_citations.txt', sep=';.-',header=None,engine='python')
sumup_overview = pd.read_csv('data/sumup_greenland.csv')

sumup_overview['citation_full'] = [sumup_citations.values[i-1] for i in sumup_overview.Citation.values]

# %% matching location
df_bav=pd.read_csv('data/csv dataset/CoreList_no_sumup.csv',sep=";",encoding='ANSI')
df_bav = df_bav.loc[df_bav.DensityAvailable=='y']
citation_bav = np.unique(df_bav.Citation)

from math import cos, asin, sqrt

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def closest(data, v):
    return min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon']))

CoordList = [{'lat':lat, 'lon':lon} for lat, lon in zip(df_bav.Northing_decdeg, df_bav.Westing_decdeg)]
CoreNo_remove = list()
for i in sumup_overview.index.values:
    Coord = {'lat': sumup_overview.Latitude[i], 'lon': sumup_overview.Longitude[i]}
    closest_coords = closest(CoordList, Coord)
    min_distance = distance(Coord['lat'],Coord['lon'],closest_coords['lat'],closest_coords['lon'])
    ind_closest = CoordList.index(closest(CoordList, Coord))
    if str(sumup_overview.date[i])[:4] == df_bav.DateCored.iloc[ind_closest][-4:]:
        if min_distance>20:
            continue
        if abs(df_bav.MaxDepthm.iloc[ind_closest]-sumup_overview.bot_depth[i]) >1:
            continue
        if sumup_overview.citation_full[i] == 'MacFerrin, M., Stevens, C., Abdalati, W., Waddington, E. (In Prep). The Firn Compaction Verification and Reconnaissance (FirnCover) dataset. In Preparation.':
            CoreNo_remove=CoreNo_remove + [df_bav.CoreNo.iloc[ind_closest]]
            continue
        if sumup_overview.citation_full[i] == 'Machguth, Horst, Mike MacFerrin, Dirk van As, et al. 2016. Greenland Meltwater Storage in Firn Limited by Near-Surface Ice Formation. Nature Climate Change 6: 390.':
            CoreNo_remove=CoreNo_remove + [df_bav.CoreNo.iloc[ind_closest]]
            continue
        if sumup_overview.citation_full[i] == 'MacFerrin, M., Machguth, H., Van As, D., Charalampidis, C., Stevens, C. M., Vandecrux, B., Heilig, A., Langen, P., Mottram, R., Fettweis, X., Van den Broeke, M.R., Moussavi, M., Abdalati, W. (In review). Rapidexpansion of Greenlandâ€™s low-permeability ice slabs in a warming climate. Nature.':
            CoreNo_remove=CoreNo_remove + [df_bav.CoreNo.iloc[ind_closest]]
            continue
        
        if sumup_overview.citation_full[i] == 'Harper, J. T., N. Humphrey, W. T. Pfeffer, J. Brown, and X. Fettweis (2012), Greenland ice-sheet contribution to sea-level rise buffered by meltwater storage in firn., Nature, 491(7423), 240â€“3, doi:10.1038/nature11566.':
            CoreNo_remove=CoreNo_remove + [df_bav.CoreNo.iloc[ind_closest]]
            continue
        if sumup_overview.citation_full[i] == 'Mosley-Thompson, et al. Local to Regional-Scale Variability of Greenland Accumulation from PARCA cores. Journal of Geophysical Research (Atmospheres), 106 (D24), 33,839-33,851. doi: 10.1029/2001JD900067':
            if str(sumup_overview.date[i])[:4] == '1998':
                CoreNo_remove=CoreNo_remove + [df_bav.CoreNo.iloc[ind_closest]]
                continue    
        if sumup_overview.Citation[i] == 191:
            continue
        if sumup_overview.Citation[i] == 190:
            continue
        # if sumup_overview.Citation[i] == 8:
        #     continue
        print(min_distance)
        print(sumup_overview.bot_depth[i] , sumup_overview.citation_full[i])
        print(df_bav.CoreNo.iloc[ind_closest],
              df_bav.MaxDepthm.iloc[ind_closest],
              df_bav.Name.iloc[ind_closest],
              df_bav.Citation.iloc[ind_closest])

        fig = plt.figure()
        plt.plot(dfg.loc[sumup_overview.coreid[i],'density']*1000,-dfg.loc[sumup_overview.coreid[i],'bot'],'x-')
        plt.plot(dfg.loc[sumup_overview.coreid[i],'density']*1000,-dfg.loc[sumup_overview.coreid[i],'mid'],'x-')
        tmp = dfg.loc[sumup_overview.coreid[i],:]
        df_core =pd.read_csv('data/csv dataset/cores/'+str(df_bav.CoreNo.iloc[ind_closest])+'.csv',
                             sep=';',header=None)
        df_core[df_core==-999]=np.nan
        plt.plot(df_core.loc[:,1],-df_core.loc[:,0]/100)
        plt.title(df_bav.loc[df_bav.CoreNo==df_bav.CoreNo.iloc[ind_closest],'Name'].values[0])      
        # plt.show()
        # plt.waitforbuttonpress()
        input()
        # TODO, give name to each sumup core, restart name when density depth decreases between two neighbors
# %%
# except
# 66
# if from 
# 'Harper, J. T., N. Humphrey, W. T. Pfeffer, J. Brown, and X. Fettweis (2012), Greenland ice-sheet contribution to sea-level rise buffered by meltwater storage in firn., Nature, 491(7423), 240â€“3, doi:10.1038/nature11566.'
# %% matching citations
from fuzzywuzzy import process, fuzz
ratio =np.zeros_like(citation_bav)
citation_sumup =np.zeros_like(citation_bav)
for i, cite in enumerate(citation_bav):
    Ratios = process.extractOne(cite,
                                sumup_citations.iloc[:,0].values.tolist(),
                                scorer=fuzz.token_set_ratio)
    ind = sumup_citations.iloc[:,0].values.tolist().index(Ratios[0])
    ratio[i] = Ratios[1]
    citation_sumup[i] = Ratios[0]


df_cit = pd.DataFrame(citation_bav,columns=['citation_bav'])
df_cit['ratio']=ratio
df_cit['citation_sumup'] = citation_sumup

#%% assessing content of matching citations
ind = [i for i, t in enumerate(df_bav.Citation.values) if t in df_cit.loc[df_cit.ratio>80,'citation_bav'].values]
dfg[dfg==-9999]= np.nan
CoreNo_remove=list()
for i in df_cit.loc[df_cit.ratio>80].index.values:
    print(i)
    print(df_cit.loc[i])
    print(np.sum(df_bav.Citation.values == df_cit.loc[i,'citation_bav']),
          np.sum(sumup_overview.citation_full.values == df_cit.loc[i,'citation_sumup']))
    
    ind_sorted_bav = np.argsort(df_bav.loc[df_bav.Citation.values == df_cit.loc[i,'citation_bav'],'MaxDepthm'].values)
    ind_sorted_sumup = np.argsort(sumup_overview.loc[sumup_overview.citation_full.values == df_cit.loc[i,'citation_sumup'],'bot_depth'].values)

    # if i in [7, 8, 55, 47, 44, 33, 25, 116, 17, 16, 15, 14, 13, 12, 11, 10, 9, 6, 4, 32]:
    #     print('Removing from bav')
    #     CoreNo_remove = CoreNo_remove + df_bav.loc[df_bav.Citation.values == df_cit.loc[i,'citation_bav'],'CoreNo'].values[ind_sorted_bav].tolist()
    #     continue
    print(np.sort(df_bav.loc[df_bav.Citation.values == df_cit.loc[i,'citation_bav'],'MaxDepthm'].values))
    
    print(np.sort(sumup_overview.loc[sumup_overview.citation_full.values == df_cit.loc[i,'citation_sumup'],'bot_depth'].values))
    print(sumup_overview.loc[sumup_overview.citation_full.values == df_cit.loc[i,'citation_sumup'],'coreid'].values)
    print('======================')
    

    ind_plot_bav = df_bav.loc[df_bav.Citation.values == df_cit.loc[i,'citation_bav'],'CoreNo'].values[ind_sorted_bav]
    ind_plot_sumup = sumup_overview.loc[sumup_overview.citation_full.values == df_cit.loc[i,'citation_sumup'],'coreid'].values[ind_sorted_sumup]
    
    max_plot = 6

    for j, k in enumerate(ind_plot_sumup):
        if j % max_plot == 0:
            fig, ax = plt.subplots(1,min(max_plot,len(ind_plot_sumup)), figsize=(min(max_plot,len(ind_plot_sumup))*4,15))
            if len(ind_plot_sumup)==1:
                ax=[ax]
        ax[j % max_plot].plot(dfg.loc[k,'density']*1000,-dfg.loc[k,'bot'],'x-')
        ax[j % max_plot].plot(dfg.loc[k,'density']*1000,-dfg.loc[k,'mid'],'x-')
        
        df_core =pd.read_csv('data/csv dataset/cores/'+str(ind_plot_bav[j-1])+'.csv',
                             sep=';',header=None)
        df_core[df_core==-999]=np.nan
        ax[j % max_plot].plot(df_core.loc[:,1],-df_core.loc[:,0]/100)
        ax[j % max_plot].set_title(df_bav.loc[df_bav.CoreNo==ind_plot_bav[j-1],'Name'].values[0])
        
# %% 