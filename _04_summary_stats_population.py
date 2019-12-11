
'''
This script pulls summary stats on the population of interest -- people with a diagnosis of chronic lung disease in the
past year, and specifically their out-patient notes.
We'll compare them in terms of
- Age (kernel density)
- Number of encounters (histogram)
- Religion (pie)
- Race (pie)
- Location (cloropleth map)
- Employment status (pie)
- Marital Status (pie)
- Sex (pie)
- Specialty (pie)
- Number of comorbidities
- Length of note (nchar) kernel density
- Length of note (number of line breaks) (kernel density)

'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import re
import geopandas as gpd
import os

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
figdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/figures/"
gisdir = "/Users/crandrew/shapefiles/"

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load "znotes" -- the notes metadata
znotes = pd.read_csv(f"{datadir}notes_metadata_2018.csv")
znotes['ENTRY_TIME'] = pd.to_datetime(znotes['ENTRY_TIME'])
znotes = znotes[znotes['ENTRY_TIME'].dt.year >= 2018]
znotes.head()

dvec = (znotes.ENTRY_TIME.dt.year-2018)*12 + znotes.ENTRY_TIME.dt.month
plt.hist(dvec)
plt.show()
# group the encounters by patient
gb_high = znotes[(znotes.highprob == True) & \
               (znotes.lowprob == False) & \
               (znotes.dxcount >=15)].groupby("PAT_ID")
gb_low = znotes[(znotes.highprob == False) & \
               (znotes.lowprob == True) & \
               (znotes.dxcount <=5)].groupby("PAT_ID")
gb_all = znotes.groupby("PAT_ID")

##############################
# map
###################
# read the shapefile
shp = gpd.read_file(f'{gisdir}tl_2015_us_zcta510.shp')
uzips = znotes[['PAT_ID', 'ZIP']].drop_duplicates()
uzips.ZIP.apply(lambda x: len(x))
uzips.head()

gb_all_zip = gb_all.agg({"ZIP": "count"})


# age
xgrid = np.linspace(18, 100, 500)
fig, ax = plt.subplots()
kde = gaussian_kde(znotes.AGE).evaluate(xgrid)
ax.plot(xgrid, kde, label = 'All')
kde_high = gaussian_kde(znotes.AGE[(znotes.highprob == True) & \
                                   (znotes.lowprob == False) & \
                                   (znotes.dxcount >=15)]).evaluate(xgrid)
ax.plot(xgrid, kde_high, label = 'High prob')
kde_low = gaussian_kde(znotes.AGE[(znotes.highprob == False) & \
                                   (znotes.lowprob == True) & \
                                   (znotes.dxcount <=5)]).evaluate(xgrid)
ax.plot(xgrid, kde_low, label = 'Low prob')
ax.legend(loc='upper left')
plt.xlabel("Age")
plt.title("Age by encounter")
plt.show()
plt.figure(figsize=(10, 10))
fig.savefig(f"{figdir}age_encounter_kd.pdf", bbox_inches='tight')

# age by unique MRN
age_gb_high = znotes[(znotes.highprob == True) & \
               (znotes.lowprob == False) & \
               (znotes.dxcount >=15)].groupby("PAT_ID").agg({"AGE":['mean']})
age_gb_low = znotes[(znotes.highprob == False) & \
               (znotes.lowprob == True) & \
               (znotes.dxcount <= 5)].groupby("PAT_ID").agg({"AGE":['mean']})
age_gb_all = znotes.groupby("PAT_ID").agg({"AGE":['mean']})
xall = age_gb_all.AGE.values
xall = xall[np.where(xall < 100)]
kde = gaussian_kde(xall).evaluate(xgrid)
xlow = age_gb_low.AGE.values
xlow = xlow[np.where(xlow < 100)]
kde_low = gaussian_kde(xlow).evaluate(xgrid)
xhigh = age_gb_high.AGE.values
xhigh = xhigh[np.where(xhigh < 100)]
kde_high = gaussian_kde(xhigh).evaluate(xgrid)

xgrid = np.linspace(18, 100, 500)
fig, ax = plt.subplots()
ax.plot(xgrid, kde, label = 'All')
ax.plot(xgrid, kde_high, label = 'High prob')
ax.plot(xgrid, kde_low, label = 'Low prob')
ax.legend(loc='upper left')
plt.xlabel("Age")
plt.title("Age by unique patient")
plt.show()
plt.figure(figsize=(10, 10))
fig.savefig(f"{figdir}age_patient_kd.pdf", bbox_inches='tight')


# noticing that I can make this simpler by just having gb objects...
gb_high = znotes[(znotes.highprob == True) & \
               (znotes.lowprob == False) & \
               (znotes.dxcount >=15)].groupby("PAT_ID")
gb_low = znotes[(znotes.highprob == False) & \
               (znotes.lowprob == True) & \
               (znotes.dxcount <=5)].groupby("PAT_ID")
gb_all = znotes.groupby("PAT_ID")

# number of encounters
xall = gb_all.agg({"AGE":'count'}).AGE.values
xlow = gb_low.agg({"AGE":'count'}).AGE.values
xhigh = gb_high.agg({"AGE":'count'}).AGE.values

xgrid = np.linspace(0, 1, 101)
fig, ax = plt.subplots()
ax.plot(xgrid, np.log10(np.quantile(xall, xgrid)), label = 'All')
ax.plot(xgrid, np.log10(np.quantile(xhigh, xgrid)), label = 'High')
ax.plot(xgrid, np.log10(np.quantile(xlow, xgrid)), label = 'Low')
plt.axhline(y=1, color='black', linestyle='-', linewidth =.5)
plt.axhline(y=2, color='black', linestyle='-', linewidth =.5)
ax.legend(loc='upper left')
plt.xlabel("Quantile")
plt.ylabel("Number of encounters (log10 scale)")
plt.title("Number of encounters")
plt.show()
fig.savefig(f"{figdir}encounters.pdf", bbox_inches='tight')

# # religion
# xall = gb_all.agg({"RELIGION":'unique'}).RELIGION.value_counts()
# xhigh = gb_high.agg({"RELIGION":'unique'}).RELIGION.value_counts()
# xlow = gb_low.agg({"RELIGION":'unique'}).RELIGION.value_counts()
#
# i = xhigh
# labels = i.index
# sizes = i.values
# # colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# # explode = (0.1, 0, 0, 0)  # explode 1st slice
#
# # Plot
# plt.pie(sizes, labels=labels,
# autopct='%1.1f%%', shadow=True, startangle=140)
#
# plt.axis('equal')
# plt.show()

# race
xall = gb_all.agg({"RACE":'unique'}).RACE.value_counts()
xhigh = gb_high.agg({"RACE":'unique'}).RACE.value_counts()
xlow = gb_low.agg({"RACE":'unique'}).RACE.value_counts()

j = 0
labs = ["all", "high", "low"]
fig, ax = plt.subplots()
for i in [xall, xhigh, xlow]:
    j+=1
    plt.subplot(1,3,j)
    labels = i.index
    sizes = i.values
    # Plot
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title(labs[j-1])
    plt.axis('equal')
plt.figure(figsize=(10, 3))
plt.show()

fig.savefig(f"{figdir}race.pdf", bbox_inches='tight', )

# # marital status
# xall = gb_all.agg({"MARITAL_STATUS":'unique'}).MARITAL_STATUS.value_counts()
# xhigh = gb_high.agg({"MARITAL_STATUS":'unique'}).MARITAL_STATUS.value_counts()
# xlow = gb_low.agg({"MARITAL_STATUS":'unique'}).MARITAL_STATUS.value_counts()
#
# j = 1
# labs = ["all", "high", "low"]
# fig, ax = plt.subplots()
# plt.subplot(1, 3, j)
# for i in [xall, xhigh, xlow]:
#     labels = i.index
#     sizes = i.values
#     print(i)
#     # Plot
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
#     plt.title(labs[j-1])
#     plt.axis('equal')
#     j+=1
# plt.figure(figsize=(100, 30))
# plt.show()
#
# fig.savefig(f"{figdir}marital.pdf", bbox_inches='tight', )

# note text length
notes = pd.read_csv(f'{datadir}note_text.csv', index_col=False)

n_char = notes.NOTE_TEXT.apply(lambda x: len(x))
n_lines = notes.NOTE_TEXT.apply(lambda x: len(re.findall('\\n', x)))
n_words = notes.NOTE_TEXT.apply(lambda x: len(re.split(' |\\n', x)))

fig, ax = plt.subplots()
plt.hist(n_char, edgecolor='black', range = [0, 100000])
plt.xlabel("number of characters per note")
plt.show()
fig.savefig(f"{figdir}nchar.pdf", bbox_inches='tight', )

fig, ax = plt.subplots()
plt.hist(n_lines, edgecolor='black', range = [0, 4000])
plt.xlabel("number of line breaks per note")
plt.show()
fig.savefig(f"{figdir}nlines.pdf", bbox_inches='tight', )

fig, ax = plt.subplots()
plt.hist(n_words, edgecolor='black', range = [0, 14000])
plt.xlabel("number of words per note")
plt.show()
fig.savefig(f"{figdir}nwords.pdf", bbox_inches='tight', )


