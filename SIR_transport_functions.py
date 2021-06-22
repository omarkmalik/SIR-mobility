import numpy as np
import pandas as pd
import random
import copy
from datetime import datetime, date, timedelta
import os
import pickle


def sir_transport(gamma, beta, n_q, T, t_delta, f_t, p, p_jce, init,  gamma_t = 0, gamma_ = 0):
    '''
    Parameters:
        gamma: a dictionary with length equal to the number of different 
               healthcare options (index 'e'). Each entry is 1/(time of infection).
               If gamma_t, gamma_ != 0, this is interpreted as the time until onset of disease.
               Otherwise, it is interpeted as the recovery rate.
        n_q: the number of states for each subpopulation
             that will quarantine. This number must be <= len(gamma)
        beta: a dictionary containing the infection rate of each behavior pattern (index 'c')
        T: the total time of the simulation.
        t_delta: the time step of the simulation
        p: a dictionary containing the fractions of the different subpopulation
        p_jce: a dictionary containing the fraction of each subpopulation j that
               behaves as c and has access to healthcare of type e
        init: the fraction of the total population that is infected at t = 0.
        f: a dictionary containing the fractions of mixing between different subpopulations for each time step
        gamma_t: The rate of testing. Only relevant when studying the quarantine state. 
        gamma_: The recovery rate from the infection. Only relevant when studying the quarantine state
    Output:
        The function outputs a dictionary that contains the different variables
        of the SIR model and their evolution over time.
    
    '''
    

    s = dict.fromkeys(p_jce)
    i = dict.fromkeys(p_jce)
    q = dict.fromkeys(p_jce)
    q_t = dict.fromkeys([j for j in p])
    r = dict.fromkeys(i.keys())
    r_t = dict.fromkeys([j for j in p])
    
    steps = int(T/t_delta)
    
    for j,c,e in p_jce:
        
        q_t[j] = [0]*steps
        r_t[j] = [0]*steps
        s[(j, c, e)] = [0]*steps
        i[(j, c, e)] = [0]*steps
        r[(j, c, e)] = [0]*steps
        q[(j, c, e)] = [0]*steps
        
        i[(j, c, e)][0] = p_jce[(j, c, e)]*init
        s[(j, c, e)][0] = p_jce[(j, c, e)]*(1 - init)
    
    new_i = copy.deepcopy(i)
    
    f = f_t[0]
    f_minus = {}
    f_plus = {}
    for j in p:
        f_minus[j] = 0
        f_plus[j] = 0
        for l in p:
            if j == l:
                continue
            f_minus[j] += f[(j, l)]
            f_plus[j] += f[(l, j)]
    flow_hist = []
    for t in range(steps-1):
        t_ = int(t_delta*t)
        if t_delta*t == t_:
            f = f_t[t_]
            f_minus = {}
            f_plus = {}
            for j in p:
                f_minus[j] = 0
                f_plus[j] = 0
                for l in p:
                    if j == l:
                        continue
                    f_minus[j] += f[(j, l)]
                    f_plus[j] += f[(l, j)]
        bflow = {}
        i_S = {}
        i_plus = {}
        i_j_ = {}
        for j, c, e in p_jce:
            if p[j] == 0:
                i_S[(j, c, e)] = 0
            else:
                i_S[(j, c, e)] = i[(j, c, e)][t]*(p[j] - f_minus[j])/p[j]
            i_plus[(j, c, e)] = 0
            for l in p:
                if p[l] != 0:
                    i_plus[(j, c, e)] += i[(l, c, e)][t]*f[(l, j)]/p[l]
                if p[j] == 0:
                    i_j_[(j, c, e, l)] = 0
                else:
                    i_j_[(j, c, e, l)] = i[(j, c, e)][t]*f[(j, l)]/p[j]
        
        for j, c, e in p_jce:
            bflow[(j, c, e)] = 0
            if p[j] != 0:
                for l in p:
                    bflow[(j, c, e)] += beta[l][c][t_]*(i_j_[(j, c, e, l)] + i_S[(l, c, e)])*f[(j, l)]/p[j]
                bflow[(j, c, e)] += beta[j][c][t_]*(i_S[(j, c, e)] + i_plus[(j, c, e)])*(p[j] - f_minus[j])/p[j]
            
            
            
            s[(j, c, e)][t+1] = s[(j, c, e)][t]*(1 - bflow[(j, c, e)]*t_delta)
            i[(j, c, e)][t+1] = i[(j, c, e)][t] + (bflow[(j, c, e)]*s[(j, c, e)][t]
                                                  - i[(j, c, e)][t]*gamma[e] - i[(j, c, e)][t]*gamma_t)*t_delta
            new_i[(j, c, e)][t+1] = bflow[(j, c, e)]*s[(j, c, e)][t]*t_delta
            
            if e < n_q:
                q[(j, c, e)][t + 1] = q[(j, c, e)][t] + gamma[e]*(
                            i[(j, c, e)][t] - q[(j, c, e)][t]*(gamma_/(gamma[e] - gamma_)))*t_delta
                r[(j, c, e)][t + 1] = r[(j, c, e)][t] + q[(j, c, e)][t]*(
                            gamma_*gamma[e]/(gamma[e] - gamma_))*t_delta
            else:
                r[(j, c, e)][t + 1] = r[(j, c, e)][t] + gamma[e]*i[(j, c, e)][t]*t_delta
                
            q_t[j][t + 1] += q[(j, c, e)][t + 1]
            r_t[j][t + 1] += r[(j, c, e)][t + 1]
        
        flow_hist.append(bflow)
            
        
    res = {'t': np.arange(0, T, t_delta), 'p': p, 'p_jce': p_jce, 'beta': beta,
                'f': f, 's':s, 'i': i, 'r': r, 'q': q, 'gamma': gamma, 'q_t': q_t,
                'r_t': r_t, 'gamma_t': gamma_t, 'b_flow' : flow_hist, 'new_i' : new_i}
    return res                               

'''The following functions perform various data handling tasks'''

def calculate_inter_region_travel(departures, arrivals):
    ''' This function uses Bayesian probability to estimate inter-borough travel.
    The values for inter_region_travel and p_d_o are taken from an MTA survey'''
    
    conditional_arrivals = {}
    conditional_departures = {}
    inter_region_travel = {0:{0: 188913, 1: 10318, 2: 7044, 3: 14926, 4: 0, 5: 7415},
                           1:{0: 165177, 1: 25597, 2: 2317, 3: 22582, 4: 0, 5: 11238},
                           2:{0: 77931, 1: 4036, 2: 18028, 3: 7763, 4: 0, 5: 1076},
                           3:{0: 245239, 1: 19803, 2: 7215, 3: 59474, 4: 0, 5: 2149},
                           4:{0: 9105, 1: 781, 2: 0, 3: 174, 4: 1135, 5: 68}}
    p_o = {}
    p_d = {}
    p_d_o = {0:{0: 0.85, 1: 0.03, 2: 0.03, 3: 0.05, 4: 0.00, 5: 0.05},
             1:{0: 0.49, 1: 0.29, 2: 0.02, 3: 0.10, 4: 0.00, 5: 0.10},
             2:{0: 0.49, 1: 0.06, 2: 0.36, 3: 0.05, 4: 0.00, 5: 0.05},
             3:{0: 0.51, 1: 0.06, 2: 0.02, 3: 0.37, 4: 0.03, 5: 0.03},
             4:{0: 0.37, 1: 0.03, 2: 0.00, 3: 0.18, 4: 0.39, 5: 0.03}}
    
    p_o_d = {0:{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             1:{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             2:{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             3:{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
             4:{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
    
    tot_trips = 0
    for j in inter_region_travel:
        p_o[j] = 0
        p_o[j] = sum(list(inter_region_travel[j].values()))
        tot_trips += p_o[j]
        
        for j_ in inter_region_travel[j]:
            if j_ in p_d:
                p_d[j_] += inter_region_travel[j][j_]
            else:
                p_d[j_] = inter_region_travel[j][j_]
    for j in p_o:
        p_o[j] *= 1/tot_trips
        p_d[j] *= 1/tot_trips
    p_d[5] *= 1/tot_trips
    for j in p_o:
        for j_ in p_o:
            p_o_d[j][j_] = p_d_o[j_][j]*p_o[j]/p_d[j_]
    for day in arrivals:
        conditional_arrivals[day] = {}
        conditional_departures[day] = {}
        for j in arrivals[day]:
            conditional_arrivals[day][j] = dict.fromkeys(range(5))
            conditional_departures[day][j] = dict.fromkeys(range(5))
            for j_ in conditional_arrivals[day][j]:
                conditional_arrivals[day][j][j_] = {}
                conditional_departures[day][j][j_] = {}
                for t in arrivals[day][j]:                    
                    conditional_arrivals[day][j][j_][t] = arrivals[day][j][t] * p_o_d[j][j_]
                for t in departures[day][j]:
                    conditional_departures[day][j][j_][t] = departures[day][j][t] * p_d_o[j][j_]
            
    return conditional_departures, conditional_arrivals         
        

def calculate_mixing_times(departures, arrivals):
    '''This function calculates the estimated average time spent by residents of one borough
    in another borough'''
    conditional_departures, conditional_arrivals = calculate_inter_region_travel(departures, arrivals)
    avg_travel_times = dict.fromkeys(range(5))
    for j in avg_travel_times:
        avg_travel_times[j] = dict.fromkeys(range(5))
        for j_ in avg_travel_times[j]:
            avg_travel_times[j][j_] = 0
    counter = copy.deepcopy(avg_travel_times)
    for day in conditional_departures:
        for j in conditional_departures[day]:
            for j_ in conditional_departures[day][j]:
                d = copy.deepcopy(conditional_departures[day][j][j_])
                a = copy.deepcopy(conditional_arrivals[day][j][j_])
                if len(a) == 0 or len(d) == 0:
                    continue
                time_a = list(a.keys())
                time_d = list(d.keys())
                time_a.sort(reverse=True)
                time_d.sort()
                temp2 = 0
                counter = 0
                for t in range(min(len(time_d), len(time_a))):
                    if time_d[t] >= time_a[t]:
                        break
                    
                    temp_d = d[time_d[t]] #departures
                    temp_a = a[time_a[t]] #arrivals
                    residue = temp_d - temp_a
                    if residue < 0:
                        if t < (len(time_d) - 1):
                            d[time_d[t + 1]] += abs(residue)
                    elif residue > 0:
                        if t < (len(time_a) - 1):
                            a[time_a[t + 1]] += residue
                        
                    temp2 += (time_a[t] - time_d[t])*min(temp_d, temp_a)
                    counter += min(temp_d, temp_a)
                if counter == 0:
                    continue
                avg_travel_times[j][j_] += temp2 / (counter * len(conditional_departures) * 24)
    return avg_travel_times

def calculate_inter_region_mixing(data_path, p, inter_region_travel):
    '''This function calculates the estimated fraction of the population of one
    borough that spends time in another borough, based on the estimated average time
    spent by residents of one borough in another borough'''
    
    turnstile_files = os.listdir(data_path)
    daily_departure = {}
    daily_arrival = {}
    borough_to_index = {'Manhattan': 0, 'Queens':1, 'Bronx':2, 'Brooklyn':3, 'Staten':4, 'Staten Island':4}
    for file in turnstile_files:
        file_path = data_path + file
        if file_path[-4:] != '.pkl':
            continue
        temp = pickle.load(open(file_path, 'rb'))
        for day in temp:
            if day not in daily_departure:
                daily_departure[day] = dict.fromkeys(range(5))
                daily_arrival[day] = dict.fromkeys(range(5))
                for j in daily_departure[day]:
                    daily_departure[day][j] = {}
                    daily_arrival[day][j] = {}
                
            for key in temp[day]:
                key_split = key.split(' ')
                t = int(float(key_split[0]))
                j = borough_to_index[key_split[1]]
                if temp[day][key] != 0:
                    if 'arrival' in file:
                        if t not in daily_arrival[day][j]:
                            daily_arrival[day][j][t] = 0
                        daily_arrival[day][j][t] += temp[day][key]
                    else:
                        if t not in daily_departure[day][j]:
                            daily_departure[day][j][t] = 0
                        daily_departure[day][j][t] += temp[day][key]
    
    avg_travel_times = calculate_mixing_times(daily_departure, daily_arrival)
    f_t = [] # a list of dictionaries where every entry is for a different time step an each entry is of size len(p) x len(p). f[(i, j)] is the fraction of borough i that visits borough j                
    for d in inter_region_travel:
        f = {(x, y): 0 for x in p for y in p}
        for i in p:
            for j in p:
                if i == -1 or j == -1:
                    continue
                f[(i, -1)] += avg_travel_times[i][j]*p[i]*inter_region_travel[d].iloc[i][j]
        f_t.append(f)
    return f_t

def calculate_mobility(data_turnstile_folder, data_covid_folder, t_len = 21):
    '''This function loads pre-processed data on the number of trips taken by residents of one
    borough to another borough. This data is then used to calculate the normalized mobility parameter
    for each borough. The function also loads in the number of new daily infections in NYC, and creates
    a training and testing split.'''
    d = date(2020, 1, 1)
    t_delta = timedelta(days = 1)
    d_end = date(2020, 12, 25)
    data_mobility = {}
    while d <= d_end:
        temp_1 = f"{d.month}".zfill(2) + "-" + f"{d.day}".zfill(2) + f"-{d.year}-morning_fraction.csv"
        temp_2 =  f"{d.month}".zfill(2) + "-" + f"{d.day}".zfill(2) + f"-{d.year}-night_fraction.csv"
        
        f_m = data_turnstile_folder / temp_1
        f_n = data_turnstile_folder / temp_2
        
        if os.path.exists(f_m) and os.path.exists(f_n):
            d_m = pd.read_csv(open(f_m, 'r'))
            d_m = d_m.rename(columns={'Unnamed: 0': 'Borough', 'Manhattan': 0, 'Queens': 1, 
                                'Bronx': 2, 'Brooklyn': 3, 'Staten Island': 4})
            d_n = pd.read_csv(open(f_n, 'r'))
            d_n = d_n.rename(columns={'Unnamed: 0': 'Borough', 'Manhattan': 0, 'Queens': 1, 
                                'Bronx': 2, 'Brooklyn': 3, 'Staten Island': 4})
            d_temp = pd.concat([d_m, d_n])
            d_temp = d_temp.groupby(level=0).mean()
            
            data_mobility[d] = d_temp
        else:
            d_temp = pd.concat([d_m, d_n])
            d_temp = d_temp.groupby(level=0).mean()
            
            data_mobility[d] = d_temp
              
        d += t_delta

    
    f = data_covid_folder / "cases-by-day.csv"
    
    data_cases = pd.read_csv( open(f, 'r'))

    

    dates = data_cases['date_of_interest']
    dates = [datetime.strptime(d, '%m/%d/%Y') for d in dates]
    dates = [date(d.year, d.month, d.day) for d in dates]
    data_cases = data_cases.assign(date_of_interest=dates)
    
    
    data_cases = data_cases[data_cases['date_of_interest'].isin(data_mobility)]
    data_mobility = {d:data_mobility[d] for d in data_cases['date_of_interest']}
    
    
    data_cases = data_cases[data_cases['date_of_interest'].isin(data_mobility)]
    
    dates = data_cases['date_of_interest']
    
    data_train = data_cases[data_cases['date_of_interest'].isin(dates[:-t_len])]
    data_test = data_cases[data_cases['date_of_interest'].isin(dates[-t_len:])]
    
    data_train = data_train[data_train['date_of_interest'].isin(data_mobility)]
    data_test = data_test[data_test['date_of_interest'].isin(data_mobility)]
    
    data_train = data_train[data_train['date_of_interest'].isin(data_mobility)]
    data_test = data_test[data_test['date_of_interest'].isin(data_mobility)]
    
    data_mobility_train = {d:data_mobility[d] for d in data_train['date_of_interest']}
    data_mobility_test = {d:data_mobility[d] for d in data_test['date_of_interest']}
    
    mobility = []
    for d in data_mobility:
        mobility.append(sum(list(data_mobility[d].sum())))
        
    m = {}
    for j in range(5):
        max_m = max([sum([data_mobility[d].iloc[j][i] for i in data_mobility[d]]) for d in data_mobility_train])
        m[j] = [sum([data_mobility[d].iloc[j][i] for i in data_mobility[d]])  for d in data_mobility]
        m[j] = [x/max_m for x in m[j]]
        
    return m, data_mobility, data_train, data_test

def calculate_beta(b_h, b_l, b_hT, b_lT, m, tau, d_cutoff):
    '''This function calculates the time dependent infection rate for each borough (index 'j')
    and behavior pattern (index 'c', which is only 0 in this case). Since we are only studying
    one behavior pattern, the second index is always 0. The infection rate within a borough depends
    upon the mobility parameter and the delay parameter, while the infection rate for the fictious 
    transport node is constant. There are different values of the infection rate before and after
    the lockdown date, indicated by the index 'd_cutoff'.'''
    beta = {}
    for j in range(-1, 5):
        if j != -1:
            beta[j] = {0: [0]*len(m[j])}
            for t in range(len(m[j])):
                if t < d_cutoff:
                    beta[j][0][t] = b_h * m[j][max(0, t - tau)]
                else:
                    beta[j][0][t] = b_l * m[j][max(0, t - tau)]
        else:
            beta[j] = {0: [0]*len(m[0])}
            for t in range(len(m[0])):
                if t < d_cutoff:
                    beta[j][0][t] = b_hT
                else:
                    beta[j][0][t] = b_lT
        
        beta[j][0] = [sum(beta[j][0][max(0, i-7):i])/len(beta[j][0][max(0, i-7):i]) for i in range(1, len(beta[j][0]) + 1)]
        
    return beta
            
    