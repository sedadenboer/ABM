import mesa
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import Political_spectrum

from visualize_beliefs import get_output_path


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
    path = get_output_path()
    data.to_csv(
        f"{path}/experiments/saved_data/pol_rep={repeats}_width={width}_net={network_type}_gridpref={grid_preference}.csv",
        index=False
        )

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data, palette=['royalblue'], errorbar=('ci', 95), legend=False)
    plt.xlabel('timestep')
    plt.ylabel('polarisation')
    
    plt.savefig(
        f"{path}/experiments/pol_rep={repeats}_width={width}_net={network_type}_gridpref={grid_preference}.png",
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
    path = get_output_path()
    network_comparison_df.to_csv(
        f"{path}/experiments/comparison_networks_rep={repeats}_width={width}_gridpref={grid_preference}.csv",
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
    plt.ylabel('polarisation')
    plt.legend(title='Network type')
    plt.savefig(f"{path}/experiments/comparison_networks_rep={repeats}_width={width}_gridpref={grid_preference}.png",
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
    path = get_output_path()
    compare_grid_network_df.to_csv(
        f"{path}/experiments/networks_vs_grid={repeats}_width={width}_net={network_type}.csv",
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
        f"{path}/experiments/networks_vs_grid={repeats}_width={width}_net={network_type}.png",
        dpi=400
        )


def grid_preference_vs_polarization(max_steps, width, lambd, mu, d1, d2, grid_radius, both_affected):
    """
    Makes lineplots with grid preference vs. polarization
    for every type of network.
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
    path = get_output_path()
    final_df.to_csv(
        f"{path}/experiments/saved_data/networks_vs_grid={repeats}_width={width}_net={network_type}.csv",
        index=False
        )

    # plotting
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(data=final_df, x='grid preference', y='polarization', hue='network',
                 palette=['royalblue', 'coral', 'mediumaquamarine', 'plum'], errorbar=('ci', 95))
    plt.legend(title='Network type')
    plt.savefig(f"{path}/experiments/grid_pref_vs_polarization_rep={repeats}_width={width}.png",
                dpi=400)


def compare_d1_d2(max_steps, repeats, width, lambd, mu,
                  network_type, grid_preference, grid_radius, both_affected):
    """
    Makes a polarization heatmap for d1 against d2.
    """
    # range of d1
    d1_vals = np.arange(0, np.sqrt(2), 0.1)
    final_df = []

    # again a lot of nested loops
    for d1 in d1_vals:
        # define d2 values
        d2_vals = np.arange(d1, np.sqrt(2), 0.1)
        for d2 in d2_vals:
            mean_polarization = []
            for _ in range(repeats):
                # compute final polarization value for
                polarization = final_polarization_value(max_steps, width, lambd, mu, d1, d2, network_type,
                                                        grid_preference, grid_radius, both_affected)
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
    print(final_df)

    # save data
    final_df.to_csv(
        f"../output_files/experiments/saved_data/compare_d1_d2={repeats}_width={width}_mu={mu}_lambd={lambd}.csv",
        index=False
        )

    # plotting heatmap for polarization
    plt.figure()
    sns.set_style("whitegrid")
    result = final_df.pivot(index='d2', columns='d1', values='mean polarization')
    sns.heatmap(result)
    plt.gca().invert_yaxis()
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"../output_files/experiments/compare_pol_d1_d2={repeats}_width={width}_mu={mu}_lambd={lambd}.png",
                dpi=400)

    # plotting heatmap for variance
    plt.figure()
    sns.set_style("whitegrid")
    result = final_df.pivot(index='d2', columns='d1', values='variance')
    sns.heatmap(result, cmap="BuPu")
    plt.gca().invert_yaxis()
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"../output_files/experiments/compare_var_d1_d2={repeats}_width={width}_mu={mu}_lambd={lambd}.png",
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
    network_comparison(max_steps, repeats, width, lambd, mu, d1, d2,
                       grid_preference, grid_radius, both_affected)
  
    # # INFLUENCE OF NETWORK VS. GRID FOR SOME GRID PREFERENCE VALUES
    # comparison_grid_network(max_steps, repeats, width, lambd, mu, d1, d2,
    #                         network_type, grid_radius, both_affected)

<<<<<<< HEAD
    # GRID PREFERENCE VS POLARIZATION FOR ALL NETWORK TYPES
    #grid_preference_vs_polarization(max_steps, width, lambd, mu, d1, d2, grid_radius, both_affected)
=======
    # # GRID PREFERENCE VS POLARIZATION FOR ALL NETWORK TYPES
    # grid_preference_vs_polarization(max_steps, width, lambd, mu, d1, d2, grid_radius, both_affected)

    # VARYING D1 AND D2
    compare_d1_d2(max_steps, repeats, width, lambd, mu,
                  network_type, grid_preference, grid_radius, both_affected)
>>>>>>> 37533c69f50b6274f60d1472bb369d0c4ecc9ec8

    # PAIRPLOT?
    #TODO

    plt.show()
