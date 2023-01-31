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


def final_polarization_value(max_steps, width, lambd, mu, d1, d2,
                             network_type, grid_preference, grid_radius, both_affected):

    # get data
    results = get_polarization(max_steps, width, lambd, mu, d1, d2,
                                network_type, grid_preference, grid_radius, both_affected)

    # get final polarization value
    polarization = results[-1]['polarization']
    
    # for the visual output
    print()

    return polarization
    

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

    # save data 
    data.to_csv(
        f"..output_files/experiments/saved_data/pol_rep={repeats}_width={width}_net={network_type}_gridpref={grid_preference}.csv",
        index=False
        )

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data, palette=['royalblue'], errorbar=('ci', 95), legend=False)
    plt.xlabel('timestep')
    plt.ylabel('polarisation')
    plt.savefig(
        f"../output_files/experiments/pol_rep={repeats}_width={width}_net={network_type}_gridpref={grid_preference}.png",
        dpi=400
        )


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

    # save data 
    network_comparison_df.to_csv(
        f"../output_files/experiments/comparison_networks_rep={repeats}_width={width}_gridpref={grid_preference}.csv",
        index=False
        )

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

    # save data 
    compare_grid_network_df.to_csv(
        f"../output_files/experiments/networks_vs_grid={repeats}_width={width}_net={network_type}.csv",
        index=False
        )

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(compare_grid_network_df,
                 palette="rocket", errorbar=('ci', 95))
    plt.xlabel('timestep')
    plt.ylabel('polarisation')
    plt.legend(title='Grid preference')
    plt.savefig(
        f"../output_files/experiments/networks_vs_grid={repeats}_width={width}_net={network_type}.png",
        dpi=400
        )


def grid_preference_vs_polarization(max_steps, width, lambd, mu, d1, d2, grid_radius, both_affected):
    """
    """
    network_types = ['BA', 'idealised', 'erdos-renyi', 'complete']

    # set range of grid preference probabilities
    grid_preference = list(np.arange(0, 1, 0.05))

    final_df = []

    # a lot of nested loops
    for network in network_types:
        for _ in range(repeats):
            for preference in grid_preference:
                # compute final polarization value for
                polarization = final_polarization_value(max_steps, width, lambd, mu, d1, d2,
                                                        network_type, grid_preference, grid_radius, both_affected)
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
    final_df.to_csv(
        f"../output_files/experiments/saved_data/networks_vs_grid={repeats}_width={width}_net={network_type}.csv",
        index=False
        )

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data=final_df, x='grid preference', y='polarization', hue='network',
                 palette=['royalblue', 'coral', 'mediumaquamarine', 'plum'], errorbar=('ci', 95))
    plt.legend(title='Network type')
    plt.savefig(f"../output_files/experiments/grid_pref_vs_polarization_rep={repeats}_width={width}.png",
                dpi=400)


if __name__ == "__main__":
    # set default parameters
    max_steps=100 # should be a multiple of 10!
    repeats=20
    width=20
    lambd=0.05
    mu=0.20
    d1=0.35
    d2=1.5
    network_type="BA"
    grid_preference=0.5
    grid_radius=2
    both_affected=True

    # # BASIC POLARIZATION TREND
    # single_polarization_trend(max_steps, repeats, width, lambd, mu, d1, d2,
    #                          network_type, grid_preference, grid_radius, both_affected)

    # # COMPARE DIFFERENT NETWORK TYPES
    # network_comparison(max_steps, repeats, width, lambd, mu, d1, d2,
    #                   grid_preference, grid_radius, both_affected)
  
    # # INFLUENCE OF NETWORK VS. GRID
    # comparison_grid_network(max_steps, repeats, width, lambd, mu, d1, d2,
    #                         network_type, grid_radius, both_affected)

    # GRID PREFERENCE VS POLARIZATION FOR ALL NETWORK TYPES
    grid_preference_vs_polarization(max_steps, width, lambd, mu, d1, d2, grid_radius, both_affected)

    # PAIRPLOT?
    #TODO

    plt.show()
