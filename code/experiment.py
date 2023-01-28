import mesa
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from model import Political_spectrum

def get_polarization():
    params = {
            "width": 10,
            "height": 10,
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
        max_steps=20,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )

    return results


def polarization_trend(repeats):
    
    polarization_dict = {}

    for count in range(repeats):
        results = get_polarization()
        print()
        for dict in results:
            polarization = dict['polarization']

            if f'experiment {count}' in polarization_dict:
                polarization_dict[f'experiment {count}'].append(polarization)
            else:
                polarization_dict[f'experiment {count}'] = [polarization]
    
    df = pd.DataFrame(polarization_dict).dropna()
    print(df)

    return df


def visualise_polarization(data):
    sns.lineplot(data=data)
    plt.savefig("../output_files/experiments/lineplot_polarization.png")

if __name__ == "__main__":

    data = polarization_trend(3)
    visualise_polarization(data)