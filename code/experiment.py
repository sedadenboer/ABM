# experiment.py
#
# Course: ABM (2022/2023)
#
# Description: Contains functions for all of the conducted experiments

import mesa
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classes.model import Political_spectrum
from save_results.output import get_output_path
from typing import List


def get_polarization(max_steps: int, lambd: float, mu: float, d1: float, d2: float,
                     network_type: str, grid_preference: float):
    """
    Runs political spectrum simulation and gets batchrunner results.

    Args:
        max_steps (int): Maximum number of time steps for the model to run.
        lambd (float, optional): Lambda, determines the distance in belief
            moved when two agents move away from each other.
            Defaults to 0.5.
        mu (float, optional): Mu, determines the distance in belief moved
            when two agents move towards each other. Defaults to 0.20.
        d1 (float, optional): The distance within which two agents will
            assimilate. Between 0 and the square root of two. Defaults to
            0.7.
        d2 (float, optional): The distance from which two agents will
            contrast. Should be larger than d1, but up to the square root
            of 2. Defaults to 1.0.
        network_type (int | str, optional): The network used to simulate
            social media. Can be "BA", "idealised", "erdos-renyi" and
            "complete". Defaults to "BA".
        grid_preference (float, optional): When a float is provided, this
            chance will be used to determine whether an agent will connect
            with someone from the grid opposed to someone from the network.
            Defaults to None.
    """
    # set parameters, some are already set to default 
    params = {
            "width": 20,
            "lambd": lambd,
            "mu": mu,
            "d1": d1,
            "d2": d2,
            "satisfaction_distance": 0.5,
            "satisfaction_threshold": 0.0,
            "mu_norm": 0.5,
            "sigma_norm": 0.2,
            "network_type": network_type,
            "grid_preference": grid_preference,
            "grid_radius": 2,
            "grid_density": 0.95,
            "both_affected": True
            }

    results = mesa.batch_run(
        Political_spectrum,
        parameters=params,
        max_steps=max_steps,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,
    )

    return results


def final_polarization_value(max_steps: int, lambd: float, mu: float, d1: float, d2: float,
                             network_type: str, grid_preference: float):
    """
    Retrieves final polarization value of a simulation of N of steps.

    Args:
        Uses the same parameters as get_polarization()
    """
    # get data
    results = get_polarization(max_steps, lambd, mu, d1, d2, network_type, grid_preference)

    # get final polarization value
    polarization = results[-1]['polarization']
    
    # for the visual output
    print()

    return polarization
    

def polarization_trend(max_steps: int, repeats: int, lambd: float, mu: float, d1: float, d2: float,
                       network_type: str, grid_preference: float):
    """
    Extracts polarization data from multiple simulations,
    organizes the data, and puts it in a dataframe.

    Args:
        repeats (int): Number of times to repeat the simulation.
        Uses the same parameters as get_polarization()
    """
    all_data = []

    for count in range(repeats):

        # initialize dictionary to store polarization data per experiment
        polarization_dict = {'data': []}

        # get data
        results = get_polarization(max_steps, lambd, mu, d1, d2, network_type, grid_preference)

        print()

        # add data to dictionary
        for dict in results:
            polarization = dict['polarization']
            polarization_dict['data'].append(polarization)
        
        # make it a dataframe and add to list
        df = pd.DataFrame(polarization_dict)
        all_data.append(df)

    # merge all experiment data into one dataframe
    final_data = pd.concat(all_data).dropna().sort_index()

    return final_data

# ------------------------------------- EXPERIMENTS -------------------------------------

def network_comparison(max_steps: int, repeats: int, lambd: float, mu: float, 
                       d1: float, d2: float, grid_preference: float):
    """
    Makes lineplots for the polarization trend from experiment data
    for various network types.

    Args:
        repeats (int): Number of times to repeat the simulation.
        Uses the same parameters as get_polarization()
    """
    network_types = ['BA', 'idealised', 'erdos-renyi', 'complete']
    network_comparison_df = pd.DataFrame()

    # run experiments for all network types
    for network in network_types:
        data = polarization_trend(max_steps, repeats, lambd, mu, d1, d2, network, grid_preference)

        # put data in dataframe
        network_comparison_df[network] = data

    # give columns representative names for plotting later
    network_comparison_df.rename(
        columns={'BA': 'Barabási–Albert', 'idealised': 'Idealised',
                 'erdos-renyi': 'Erdős–Rényi', 'complete': 'Complete'},
        inplace=True)

    # save data and cleanup the dataframe
    path = get_output_path()
    network_comparison_df.to_csv(
        f"{path}/comparison_networks_rep={repeats}_gridpref={grid_preference}_d1d2={d1,d2}.csv",
        index=False
        )
    column = ['Barabási–Albert', 'Idealised', 'Erdős–Rényi', 'Complete']
    network_comparison_df = network_comparison_df.melt(value_vars=column, ignore_index = False)
    network_comparison_df.columns = ['network', 'pol_ind']
    network_comparison_df['step'] = network_comparison_df.index

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data = network_comparison_df.reset_index(), y = 'pol_ind', x = 'step', hue = 'network',
                 palette=['royalblue', 'coral', 'mediumaquamarine', 'plum'], ci=95)
    plt.xlabel('timestep')
    plt.ylabel('polarization')
    plt.legend(title='Network type')
    plt.savefig(f"{path}/comparison_networks_rep={repeats}_gridpref={grid_preference}.png",
                dpi=400)


def grid_preference_vs_polarization(network_types: List[str], max_steps: int, lambd: float, mu: float,
                                    d1: float, d2: float, repeats: int, stepsize: float):
    """
    Makes lineplots with grid preference vs. polarization
    for every type of network.

    Args:
        network_types (List[str]): List of all of the network types.
        repeats (int): Number of times to repeat the simulation.
        stepsize (float): Stepsize for determining the grid preference values.
        Uses the same parameters as get_polarization()
    """

    # set range of grid preference probabilities
    grid_preference = list(np.arange(0, 1 + stepsize, stepsize))

    final_df = []

    # loop over all of the varied parameters
    for network in network_types:
        for _ in range(repeats):
            for preference in grid_preference:
                # compute final polarization value for
                polarization = final_polarization_value(max_steps, lambd, mu, d1, d2, 
                                                        network_type, grid_preference)
                df = pd.DataFrame({'grid preference': [preference],
                                   'network': [network],
                                   'polarization': [polarization]})
                final_df.append(df)

    # combine all data
    final_df = pd.concat(final_df)
    formal_names = {'BA': 'Barabási–Albert', 'idealised': 'Idealised',
                    'erdos-renyi': 'Erdős–Rényi', 'complete': 'Complete'}
    final_df = final_df.replace(formal_names)

    # save data
    path = get_output_path()
    final_df.to_csv(
        f"{path}/grid_pref_vs_polarization_rep={repeats}_d1d2={d1,d2}.csv",
        index=False
        )

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data=final_df, x='grid preference', y='polarization', hue='network',
                 palette=['royalblue', 'coral', 'mediumaquamarine', 'plum'], errorbar=('ci', 95))
    plt.legend(title='Network type')
    plt.savefig(f"{path}/grid_pref_vs_polarization_rep={repeats}_d1d2={d1,d2}.png",
                dpi=400)


def compare_d1_d2(max_steps: int, repeats: int, lambd: float, mu: float,
                  network_type: str, grid_preference: float):
    """
    Makes a polarization heatmap for d1 against d2.

    Args:
        repeats (int): Number of times to repeat the simulation.
        Uses the same parameters as get_polarization()
    """
    # range of d1
    d1_vals = np.arange(0, np.sqrt(2), 0.1)
    final_df = []

    # loop over the varied parameters
    for d1 in d1_vals:
        # define d2 values
        d2_vals = np.arange(d1, np.sqrt(2), 0.1)
        for d2 in d2_vals:
            mean_polarization = []
            for _ in range(repeats):
                # compute final polarization value for
                polarization = final_polarization_value(max_steps, lambd, mu, d1, d2,
                                                        network_type, grid_preference)
                # calculate mean polarization over all repeats
                mean_polarization.append(polarization)

            # add data to dataframe
            df = pd.DataFrame({'d1': [round(d1, 2)],
                               'd2': [round(d2, 2)],
                               'mean polarization': [np.mean(mean_polarization)],
                               'variance': [np.var(mean_polarization)]})

            final_df.append(df)

    # combine all data
    final_df = pd.concat(final_df)

    # save data
    path = get_output_path()
    final_df.to_csv(
        f"{path}/compare_d1_d2={repeats}_mu={mu}_lambd={lambd}.csv",
        index=False
        )

    # plotting heatmap for polarization
    plt.figure()
    sns.set_style("whitegrid")
    result = final_df.pivot(index='d2', columns='d1', values='mean polarization')
    sns.heatmap(result, cmap='flare', cbar_kws={'label': 'mean polarization'})
    plt.xlabel(r'$d_1$', fontsize=16)
    plt.ylabel(r'$d_2$', fontsize=16)
    plt.gca().invert_yaxis()
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig(f"{path}/compare_pol_d1_d2={repeats}_mu={mu}_lambd={lambd}.png",
                dpi=400)

    # plotting heatmap for variance
    plt.figure()
    sns.set_style("whitegrid")
    result = final_df.pivot(index='d2', columns='d1', values='variance')
    sns.heatmap(result, cmap='crest', cbar_kws={'label': 'polarization variance'})
    plt.xlabel(r'$d_1$', fontsize=16)
    plt.ylabel(r'$d_2$', fontsize=16)
    plt.gca().invert_yaxis()
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig(f"{path}/compare_var_d1_d2={repeats}_mu={mu}_lambd={lambd}.png",
                dpi=400)


if __name__ == "__main__":
    # set default parameters
    max_steps=2 # MAKE SURE TO HAVE THE CORRECT MEASURING STEPS VALUE IN polarization() in model.py
    repeats=1
    lambd=0.05
    mu=0.20
    d1=0.1
    d2=0.2
    network_type="BA"
    grid_preference=0.5
    network_types = ["BA", "idealised", "erdos-renyi", "complete"]
    stepsize=0.05

    # EXPERIMENT 1: VARYING D1 AND D2 FOR DIFFERENT MU AND LAMBDA COMBINATIONS
    compare_d1_d2(max_steps, repeats, lambd, mu, network_type, grid_preference)

    # EXPERIMENT 2: COMPARE DIFFERENT NETWORK TYPES
    # !!! make sure to set "MEASURE_STEPS" <= 5 in model.py for this experiment !!!
    max_steps_network_comparison=200
    network_comparison(max_steps_network_comparison, repeats, lambd, mu, d1, d2, grid_preference)

    # EXPERIMENT 3: GRID PREFERENCE VS POLARIZATION FOR ALL NETWORK TYPES
    grid_preference_vs_polarization(network_types, max_steps, lambd, mu, d1, d2, repeats, stepsize)

    # grid preference vs polarization for BA network
    network_types=['BA']
    max_steps_BA=2
    stepsize=0.2
    
    # expected low polarization
    d1=1
    d2=1.4
    grid_preference_vs_polarization(network_types, max_steps_BA, lambd, mu, d1, d2, repeats, stepsize)
    
    # expected high polarization
    d1=0.2
    d2=0.4
    grid_preference_vs_polarization(network_types, max_steps_BA, lambd, mu, d1, d2, repeats, stepsize)

    plt.show()
