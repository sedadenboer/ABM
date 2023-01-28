import mesa
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from model import Political_spectrum

def get_polarization():
    params = {
            "width": 10,
            "lambd": 0.5,
            "mu": 0.20,
            "d1": 0.7,
            "d2": 1.0,
            "mu_norm": 0.5,
            "sigma_norm": 0.2,
            "network_type": "BA",
            "grid_preference": 0.5,
            "both_affected": True
            }

    results = mesa.batch_run(
        Political_spectrum,
        parameters=params,
        iterations=1,
        max_steps=100,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )

    return results


def polarization_trend(repeats):
    
    all_data = []

    for count in range(repeats):

        # initialize dictionary to store polarization data per experiment
        polarization_dict = {'data': []}

        # get data
        results = get_polarization()

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


def visualise_polarization(df):

    # make lineplot
    plt.figure()
    sns.set_style("whitegrid")
    sns.lineplot(df, palette=['royalblue'], errorbar=('ci', 95), legend=False)
    plt.xlabel('timestep')
    plt.ylabel('polarisation')

    # save and show plot
    plt.savefig("../output_files/experiments/lineplot_polarization.png", dpi=400)
    plt.show()



if __name__ == "__main__":

    data = polarization_trend(100)
    visualise_polarization(data)