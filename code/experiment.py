import mesa
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import Political_spectrum


def get_polarization(max_steps, width, lambd, mu, d1, d2,
                     network_type, grid_preference, grid_radius, both_affected):
    """
    Runs political spectrum simulation and gets batchrunner results.
    """
    params = {
            "width": width,
            "lambd": lambd,
            "mu": mu,
            "d1": d1,
            "d2": d2,
            "mu_norm": 0.5,
            "sigma_norm": 0.2,
            "network_type": network_type,
            "grid_preference": grid_preference,
            "grid_radius": grid_radius,
            "both_affected": both_affected
            }

    results = mesa.batch_run(
        Political_spectrum,
        parameters=params,
        iterations=1,
        max_steps=max_steps,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )

    return results


def polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
                       network_type, grid_preference, grid_radius, both_affected):
    """
    Extracts polarization data from multiple simulations,
    organizes the data, and puts it in a dataframe.
    """
    all_data = []

    for count in range(repeats):

        # initialize dictionary to store polarization data per experiment
        polarization_dict = {'data': []}

        # get data
        results = get_polarization(max_steps, width, lambd, mu, d1, d2,
                                   network_type, grid_preference, grid_radius, both_affected)

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



# ---------------------------- EXPERIMENTS ----------------------------

def single_polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
                              network_type, grid_preference, grid_radius, both_affected):
    """
    Makes a simple lineplot for the polarization trend from experiment data.
    """
    # get data from simulation
    data = polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
                              network_type, grid_preference, grid_radius, both_affected)

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data, palette=['royalblue'], errorbar=('ci', 95), legend=False)
    plt.xlabel('timestep')
    plt.ylabel('polarisation')
    plt.savefig(f"../output_files/experiments/pol_rep={repeats}_width={width}_net={network_type}_gridpref={grid_preference}.png",
                dpi=400)


def network_comparison(max_steps, repeats, width, lambd, mu, d1, d2,
                       grid_preference, grid_radius, both_affected):
    """
    Makes lineplots for the polarization trend from experiment data
    for various network types.
    """
    network_types = ['BA', 'idealised', 'erdos-renyi', 'complete']
    network_comparison_df = pd.DataFrame()

    # run experiments for all network types
    for network in network_types:
        data = polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
                                  network, grid_preference, grid_radius, both_affected)

        # put data in dataframe
        network_comparison_df[network] = data

    # give columns representative names for plotting later
    network_comparison_df.rename(
        columns={'BA': 'Barabási–Albert', 'idealised': 'Idealised',
                 'erdos-renyi': 'Erdős–Rényi', 'complete': 'Complete'},
        inplace=True)

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(network_comparison_df,
                 palette=['royalblue', 'coral', 'mediumaquamarine', 'plum'], errorbar=('ci', 95))
    plt.xlabel('timestep')
    plt.ylabel('polarisation')
    plt.legend(title='Network type')
    plt.savefig(f"../output_files/experiments/comparison_networks_rep={repeats}_width={width}_gridpref={grid_preference}.png",
                dpi=400)


def comparison_grid_network(max_steps, repeats, width, lambd, mu, d1, d2,
                            network_type, grid_radius, both_affected):
    """
    Makes lineplots for the polarization trend from experiment data
    for a varying grid preference. Like this, we can investigate the 
    effect of interactions with people from the network or the grid.
    """
    # set range of grid preference probabilities
    grid_preference = np.linspace(0.0, 1.0, num=5)
    compare_grid_network_df = pd.DataFrame()

    # run experiments for all grid preferences
    for preference in grid_preference:
        data = polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
                                  network_type, preference, grid_radius, both_affected)

        # put data in dataframe
        compare_grid_network_df[preference] = data

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(compare_grid_network_df,
                 palette="rocket", errorbar=('ci', 95))
    plt.xlabel('timestep')
    plt.ylabel('polarisation')
    plt.legend(title='Grid preference')
    plt.savefig(f"../output_files/experiments/networks_vs_grid={repeats}_width={width}_net={network_type}.png",
                dpi=400)



if __name__ == "__main__":
    # set parameters
    max_steps=100
    repeats=10
    width=10
    lambd=0.05
    mu=0.20
    d1=0.35
    d2=1.5
    network_type="BA"
    grid_preference=0.5
    grid_radius=2
    both_affected=True

    # BASIC POLARIZATION TREND
    single_polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
                             network_type, grid_preference, grid_radius, both_affected)

    # COMPARE DIFFERENT NETWORK TYPES
    network_comparison(max_steps, repeats, width, lambd, mu, d1, d2,
                      grid_preference, grid_radius, both_affected)
  
    # INFLUENCE OF NETWORK VS. GRID
    comparison_grid_network(max_steps, repeats, width, lambd, mu, d1, d2,
                            network_type, grid_radius, both_affected)

    # PAIRPLOT?
    #TODO
    
    plt.show()