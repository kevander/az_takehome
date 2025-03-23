import IPython
import numpy as np
import pandas as pd
import tqdm
from typing import List
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
pio.renderers.default = "browser" 

class Simulation:
    '''
    Parameters
    ----------
    n : int
        Desired number of patients.
    n_visits: int
        Desired number of visits per patient.
    r: float
        Desired correlation between baseline features and time to death
    '''
    def __init__(self,  
                 n_pbo: int = 500,
                 n_trt: int = 500,
                 n_visits: int = 10,
                 event_rate_pbo: float = 0.05,
                 event_rate_trt: float = 0.10,
                 event_rate_change_over_time_pbo: float = 0.01,
                 event_rate_change_over_time_trt: float = 0.01,
                 cv_mean_pbo: float = 100,
                 cv_sd_pbo: float = 20,
                 cv_noise_sd_pbo: float = 3,
                 cv_change_over_time_pbo: float = -10,
                 cv_mean_trt: float = 100,
                 cv_sd_trt: float = 20,
                 cv_noise_sd_trt: float = 3,
                 cv_change_over_time_trt: float = -10,
                 seed: int = 2948,
                 ):
        self.n_pbo = n_pbo
        self.n_trt = n_trt
        self.n_visits = n_visits
        self.event_rate_pbo = event_rate_pbo
        self.event_rate_trt = event_rate_trt
        self.event_rate_change_over_time_pbo = event_rate_change_over_time_pbo
        self.event_rate_change_over_time_trt = event_rate_change_over_time_trt
        self.cv_mean_pbo = cv_mean_pbo
        self.cv_sd_pbo = cv_sd_pbo
        self.cv_noise_sd_pbo = cv_noise_sd_pbo
        self.cv_std_change_over_time_pbo = cv_change_over_time_pbo
        self.cv_mean_trt = cv_mean_trt
        self.cv_sd_trt = cv_sd_trt
        self.cv_noise_sd_trt = cv_noise_sd_trt
        self.cv_std_change_over_time_trt = cv_change_over_time_trt
        self.seed = seed

    def simulate(self):
        IPython.embed()
        pass

        # Run Placebo Arm Simulation 
    
    def run_simulation(self, 
                patient_ids: List[int],
                arm_name: str,
                arm_n: int,
                event_rate: float,
                event_rate_change_rate: float,
                cv_mean: float,
                cv_sd: float,
                cv_std_change_over_time: float,
                cv_noise_sd: float):
        
        # baseline continuous variable 
        cv_baseline = np.random.normal(loc=cv_mean, 
                                        scale=cv_sd, 
                                        size=arm_n)
        
        ### C.V. Change Rates over Time
        # divide cv_std_change_over_time by n_visits to get the change at each visit
        # *cv_sd converts the change rate from SD to native units
        cv_change_at_each_visit = (cv_std_change_over_time / self.n_visits) * cv_sd

        # add some noise to the change rates, otherwise we have an unrealistic scenario
        # where all patients have the same trajectory
        cv_change_noise = np.random.normal(loc=0, 
                                        scale=cv_change_at_each_visit*0.3, 
                                        size=arm_n)
        cv_change_rates = cv_change_at_each_visit + cv_change_noise

        visit_cv_data = []
        event_data = []
        for visit in range(self.n_visits):
            ### Simulate Continuous Variable Data
            # Baseline visit
            if visit == 0:
                visit_values = cv_baseline
            else: 
                # Get the C.V. values for this visit (rate x time)
                expected_values = cv_baseline + (cv_change_rates * visit)

                # Add noise to the trajectory
                noise_terms = np.random.normal(loc=0, scale=cv_noise_sd, size=arm_n)
                visit_values = expected_values + noise_terms
            visit_cv_data.append(pd.Series(visit_values))

            ### Simulate Events
            # sample from a bernoulli distribution to determine if the patient dies
            events = np.random.binomial(n=1, p=event_rate, size=arm_n)
            event_data.append(pd.Series(events))

        # Concatenate CV/event data from each visit
        event_df    = pd.concat(event_data, axis=1)
        visit_cv_df = pd.concat(visit_cv_data, axis=1)
        
        # format the data and mask the CV measurements if the patient died
        cv_df, death_df = self.format_output(event_df, 
                                                visit_cv_df, 
                                                patient_ids, 
                                                format='long')
        cv_df.insert(1, 'arm', arm_name)
        death_df.insert(1, 'arm', arm_name)
        return cv_df, death_df
    
    
    # def simulate_old(self):
        
    #     # N visits
    #     n_visits = self.n_visits


    #     sim_data = []
    #     sim_cv_data = []
    #     sim_death_data = []
    #     for arm in ['pbo', 'trt']:
    #         if arm == 'pbo':
    #             patient_ids = np.arange(1, arms_n['pbo']+1)
    #         else:
    #             patient_ids = np.arange(arms_n['pbo']+1, arms_n['pbo']+arms_n['trt']+1)
            
    #         # baseline continuous variable 
    #         cv_baseline = np.random.normal(loc=continuous_variable[arm]['mean'], 
    #                                                 scale=continuous_variable[arm]['sd'], 
    #                                                 size=arms_n[arm])

    #         # distribution of C.V. changes over time, with some gaussian noise added
    #         cv_change_noise = np.random.normal(loc=0, 
    #                                         scale=np.abs(cv_change_rate[arm])*0.3, 
    #                                         size=arms_n[arm])
    #         cv_change_rates = cv_change_rate[arm] + cv_change_noise
    #         # truncate change rates to be between -1 and 1
    #         cv_change_rates = np.clip(cv_change_rates, -1, 1)
            
    #         # Create visits array
    #         visits_array = np.arange(n_visits)
            
    #         # Simulate visits
    #         is_alive = True

    #         # Calculate the intercept that gives the baseline probability
    #         p0 = event_rate[arm]
    #         alpha = np.log(p0 / (1 - p0))

    #         event_data = []
    #         visit_cv_data = []
    #         for visit in range(n_visits):
    #             if visit == 0:
    #                 # Baseline visit
    #                 visit_values = cv_baseline
    #             else:
    #                 # Add noise to the trajectory
    #                 expected_values = cv_baseline + (cv_change_rates * visit)
    #                 noise_terms     = np.random.normal(loc=0, 
    #                                                   scale=continuous_variable_noise_sd[arm],
    #                                                   size=arms_n[arm])
    #                 visit_values = expected_values + noise_terms
    #             visit_cv_data.append(pd.Series(visit_values))

    #             # sample from a bernoulli distribution to determine if the patient dies
    #             events = np.random.binomial(n=1, p=event_rate[arm], size=arms_n[arm])
    #             event_data.append(pd.Series(events))
            
    #         event_df    = pd.concat(event_data, axis=1)
    #         visit_cv_df = pd.concat(visit_cv_data, axis=1)

    #         cv_df, death_df = self.format_output(event_df, 
    #                                              visit_cv_df, 
    #                                              patient_ids, 
    #                                              format='long')
    #         cv_df.insert(1, 'arm', arm)
    #         death_df.insert(1, 'arm', arm)
    #         sim_cv_data.append(cv_df)
    #         sim_death_data.append(death_df)
    #     sim_cv_data = pd.concat(sim_cv_data)
    #     sim_death_data = pd.concat(sim_death_data)
    #     return sim_cv_data, sim_death_data
            
    def format_output(self, 
                      event_df = pd.DataFrame, 
                      visit_cv_df = pd.DataFrame, 
                      patient_ids = List[int],
                      format='long'):
        if format == 'long':
            # Format continuous variable data
            cv_df = visit_cv_df.copy()
            cv_df.columns = range(1, self.n_visits+1)
            cv_df.insert(0, 'patient_id', patient_ids )
            visit_cv_long = cv_df.melt(id_vars='patient_id')
            visit_cv_long.columns = ['patient_id', 'visit', 'continuous_measure']

            # for each row, find first column with a 1
            # this corresponds to the visit where a death was observed
            death_visits = event_df.idxmax(axis=1)
            death_df = pd.DataFrame({
                'patient_id': patient_ids,
                'visit': death_visits
            })
            death_df['event'] = 0
            death_df.loc[death_df['visit'] != 0, 'event'] = 1
            death_df['visit'] = death_df['visit'].replace(0, self.n_visits)

            # remove cv measurements from after death
            filter_df = visit_cv_long.merge(death_df, 
                                on=['patient_id'], 
                                suffixes=('', '_death'),
                                how='left')
            filter_df.loc[filter_df['visit'] > filter_df['visit_death'], 'continuous_measure'] = np.nan
            filter_df.drop(columns=['visit_death', 'event'], inplace=True)
            filter_df = filter_df.sort_values(['patient_id', 'visit'])
            return filter_df, death_df


    def plot_continous_variable_over_time(self, 
                                   sim_cv_data: pd.DataFrame,
                                   groups: str = None):
        
        plot_df = sim_cv_data\
            .groupby(['arm', 'visit'])['continuous_measure']\
            .agg(['mean', 'std', 'sem']).reset_index()
        

        # plotly scatter with errorbars and lines connecting each dot
        fig = go.Figure()

        for arm in plot_df['arm'].unique():
            arm_df = plot_df[plot_df['arm'] == arm]
            fig.add_trace(go.Scatter(
                x=arm_df['visit'],
                y=arm_df['mean'],
                mode='lines+markers',
                name=arm,
                error_y=dict(
                    type='data',
                    array=arm_df['sem'],
                    visible=True
                )
            ))

        fig.show()
        return fig




