
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

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/"
figdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/figures/"

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load "znotes" -- the notes metadata
znotes = pd.read_csv(f"{datadir}notes_metadata_2018.csv")
znotes['ENTRY_TIME'] = pd.to_datetime(znotes['ENTRY_TIME'])
znotes = znotes[znotes['ENTRY_TIME'].dt.year >= 2018]
znotes.head()

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
