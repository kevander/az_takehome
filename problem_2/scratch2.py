import pandas as pd
import sys
sys.path.append('/Users/kevinanderson/Projects/az_takehome/problem_2')
import problem_2.scratch as scratch
import simulation
import importlib


importlib.reload(simulation)


# Low Dependency Simulation 
low_dep_sim = simulation.Simulation(sim_name='low_dependency',
                                    event_rate=0.1,
                                    cv_mean=100, 
                                    cv_sd=20,
                                    cv_change_over_time=-3,
                                    cv_noise_sd=30, 
                                    relationship_coefficient=1
                                    )
low_dep_cv_data, low_dep_death_data = low_dep_sim.simulate()

# High Dependency Simulation
high_dep_sim = simulation.Simulation(sim_name='high_dependency',
                                    event_rate=0.01,
                                    cv_mean=100, 
                                    cv_sd=20,
                                    cv_change_over_time=-1,
                                    cv_noise_sd=1, 
                                    relationship_coefficient=0
                                    )
high_dep_cv_data, high_dep_death_data = high_dep_sim.simulate()

cv_plot_df = pd.concat([
    low_dep_cv_data,
    high_dep_cv_data
])

event_plot_df = pd.concat([
    low_dep_death_data,
    high_dep_death_data
])

fig = low_dep_sim.plot_continous_variable_over_time(cv_plot_df, event_plot_df)

low_cox_res = low_dep_sim.run_cox_timevary_cox_propotional_hazard(low_dep_cv_data, low_dep_death_data)
high_cox_res = low_dep_sim.run_cox_timevary_cox_propotional_hazard(high_dep_cv_data, high_dep_death_data)


import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter

df = high_dep_cv_data.merge(high_dep_death_data, 
                            on=['patient_id', 'visit'], how='left')
df = df.loc[df['continuous_measure'].notna()].copy()
df['event'] = df['event'].replace(np.nan, 0)
df['start'] = df['visit'] - 1

keep_cols = ['patient_id', 'visit', 'continuous_measure', 'event', 'start']
cox_df = df[keep_cols].copy()
cox_df['visit'] = cox_df['visit'].astype(int)

# Fit the time-dependent Cox model
ctv = CoxTimeVaryingFitter()
ctv.fit(cox_df, id_col="patient_id", start_col="start", stop_col="visit", event_col="event")
ctv.summary


fig = low_dep_sim.plot_continous_variable_over_time(cv_plot_df, event_plot_df)

fig.show()


