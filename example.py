#%% Initialize parameters


import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import statistics 
import pickle
import copy

import SIR_transport_functions as sir

turnstile_files = "./data/turnstile_data/" #The raw data showing the number of people arriving and departing at each borough. It is used to calculate inter-borough travel
data_turnstile_folder = Path("data/mobility/fraction") #Each file in this folder contains the number of trips from every borough to every other borough, expressed as a fraction of the population of the borough of departure
data_covid_folder = Path("data/covid_19_data") #Publicly available data about the number of newly reported covid-19 cases in NYC, reported by the NYC government

#The population of each borough       
pop_tot = 8336817
p = {0: 1628706/pop_tot, #Manhattan
      1: 2253858/pop_tot, #Queens
      2: 1418207/pop_tot, #Bronx
      3: 2559903/pop_tot, #Brooklyn
      4: 476143/pop_tot, #Staten Island
      -1: 0} #This is the transport node

p_jce = {} #the amount of each subpopulation as a fraction of the total population. Index j refers to boroughs, index c to behavior typers, and index e to healthcare type.
for j in p:
    for c in range(1):
        for e in range(1):
                p_jce[(j, c, e)] = p[j]

b_h = 1.63
b_l = 0.6
b_hT = 4 #values of infection rate before lockdown
b_lT = 4  #values of infection rate after lockdown
gamma = {0: 0.04} #values of recovery rate 
init = 1/pop_tot #values of initial infected population
tau = 22 #The delay parameter
t_len = 21
n_q = 1
d_l = date(2020, 3, 22) #Date of lockdown
t_delta = 10**(-2) #time resolution

m, inter_region_travel, data_train, data_test = sir.calculate_mobility(data_turnstile_folder, data_covid_folder, t_len)
f_t = sir.calculate_inter_region_mixing(turnstile_files, p, inter_region_travel)

dates = list(data_train['date_of_interest']) + list(data_test['date_of_interest'])
d_cutoff = dates.index(d_l) #The index of the cutoff date
T = len(data_train) + len(data_test)
beta = sir.calculate_beta(b_h, b_l, b_hT, b_lT, m, tau, d_cutoff)


i_gt_train = [x/pop_tot for x in list(data_train['CASE_COUNT_7DAY_AVG'])] #ground truth total number of cases
i_gt_test = [x/pop_tot for x in list(data_test['CASE_COUNT_7DAY_AVG'])]

res = sir.sir_transport(gamma, beta, n_q, T, t_delta, f_t, p, p_jce, init)

#Construct the total daily new cases in NYC
daily_i = [0]*T
for j, c, e in p_jce:
    for t in range(T):
        daily_i[t] += sum(res['new_i'][(j, c, e)][int(t/t_delta): int((t + 1)/t_delta)])

i_fit_train = daily_i[:-t_len]
i_fit_test = daily_i[-t_len:]

E_in = mean_squared_error(i_gt_train, i_fit_train)
E_out = mean_squared_error(i_gt_test, i_fit_test)

#%% Plot Results

xt = list(range(0, 300, 50)) + [len(i_gt_train)]
plt.xlabel('time (days)', fontsize=12)
plt.ylabel('$i^{new}$(t)', fontsize=12)
# plt.title('$R^2$ optimization')
plt.plot(range(0, T - t_len), i_gt_train, 'r')
plt.plot(range(0, T - t_len), i_fit_train, 'k')
plt.axvline(x=len(i_gt_train), color='k', linestyle='--')
plt.xticks(xt, [dates[i] for i in xt], rotation=75)
plt.plot(range(T - t_len, T), i_gt_test, color='g')
plt.plot(range(T - t_len, T), i_fit_test, 'b')
plt.show()
 