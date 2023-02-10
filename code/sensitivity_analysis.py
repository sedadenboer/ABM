# sensitivity_analysis.py
#
# Course: ABM (2022/2023)
#
# Description: File to conduct the sensitivity
# analysis for all of the parameters form the model.

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from mesa.batchrunner import batch_run
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from itertools import combinations
from classes.model import Political_spectrum
from save_results.output import get_output_path


def create_samples(problem: dict, num_samples: int, second_order: bool, save_data: bool = False, save_as: str = None):
    """Creates samples for the sobol analysis using saltelli's sampling scheme.

    Args:
        problem (dict): The problem description. Should look like:
            dict{"num_vars": int, "names": list[str], "bounds": list[list]}
            where "names" has all names of the variables and "bounds" contains
            the bounds of these variables.
        num_samples (int): The number of samples, should be a multiple of 2.
        second_order (bool): Whether to include second order analysis.
        save_data (bool, optional): Wheter to save the data. Defaults to False.
        save_as (str, optional): What to save the data as. Defaults to None.

    Returns:
        dict: The generated samples.
    """
    param_values = np.array(sobol_sample.sample(problem, num_samples, calc_second_order=second_order))
    parameters = {}
    parameters_list = []
    for run in range(len(param_values)):
        variables = {}
        for name, val in zip(problem["names"], param_values[run]):
            variables[name] = val

        # make sure the network is chosen as index
        variables["network_type"] = int(variables["network_type"])

        # convert the sample to the right type
        variables["width"] = int(variables["width"])
        variables["grid_radius"] = int(variables["grid_radius"])
        variables["both_affected"] = int(variables["both_affected"] > 0.5)

        parameters[run] = variables
        parameters_list.append(variables)

    if save_data:
        df = pd.DataFrame(parameters_list)
        path = get_output_path()
        df.to_csv(f"{path}/{save_as}_samples.csv", index=False)
        
    return parameters_list


def run_batch_model(parameters):
    """Runs the model once using mesa's batch_run.

    Args:
        parameters (list): The parameters used for this run.

    Returns:
        list[dict[str]]: The result of the batch run.
    """
    max_steps, data_collection_period, variables = parameters
    batch = batch_run(model_cls=Political_spectrum,
                        parameters=variables, # will run each combination of parameters
                        number_processes=1, # not parallel
                        iterations=1,
                        data_collection_period=data_collection_period,
                        max_steps=max_steps,
                        display_progress=False)
    return batch


def sobol_run_samples(problem: dict,
                      repeats: int,
                      max_steps: int,
                      data_collection_period: int,
                      from_data: bool,
                      save_as: str=None,
                      samples: dict=None,
                      number_processes: int = None,
                      save_data: bool = False) -> pd.DataFrame:
    """Runs the provided samples and collects the data for sobol analysis.

    Args:
        problem (dict): The problem description. Should look like:
            dict{"num_vars": int, "names": list[str], "bounds": list[list]}
            where "names" has all names of the variables and "bounds" contains
            the bounds of these variables.
        repeats (int): The number of repeats run for each sample combination.
        max_steps (int): The maximum number of steps of the model.
        data_collection_period (int): When the data is collected.
        from_data (bool): Whether to run samples that are saved as csv.
        save_as (str, optional): What the samples are stored as and what to
            save the final data as. Defaults to None.
        samples (dict, optional): The samples. Defaults to None.
        number_processes (int, optional): The number of processes for parallel
            computing. Defaults to None.
        save_data (bool, optional): Whether to save the data as a csv file.
            Defaults to False.

    Raises:
        ValueError: If no samples are provided.
        ValueError: If the number of processes is negative.

    Returns:
        pd.DataFrame: The final data after running.
    """

    if from_data:
        # load the samples
        path = get_output_path()
        df = pd.read_csv(f"{path}/{save_as}_samples.csv")
        samples = df.to_dict("records")
    else:
        # make sure samples are given
        if not samples:
            raise ValueError("Please provide samples if not run from a file.")

    parameters = [[max_steps, data_collection_period, i] for i in samples]

    # use the number of processors of this machine
    if not number_processes:
        number_processes = mp.cpu_count()

    if number_processes > 0:
        pool = mp.Pool(number_processes)

        for rep in range(repeats):
            # create a dataframe to save the data
            data = pd.DataFrame(index=range(len(samples)),
                                columns=problem["names"])
            data["polarization"] = None

            run = 0 # keep track of the runs for the dataframe

            process = pool.map_async(run_batch_model, parameters)
            result = process.get()
            for i in range(len(result)):
                run_dict = result[i][0]
                for name in data.columns:
                    data.loc[run, name] = run_dict[name]
                run += 1

            if save_data:
                path = get_output_path()
                user = os.environ.get('USER', os.environ.get('USERNAME'))
                data.to_csv(f"{path}/{save_as}_{user}_{rep}.csv", index=False)

            print(data)

        pool.close()

    else:
        raise ValueError("The number of processes should be a positive value")

    return data

def plot_index(s, params, i, title=''):    
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params  (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')
    plt.tight_layout()

def sobol_analyze_data(problem: dict, from_file: bool=True, save_as: str=None, data: pd.DataFrame=None, second_order:bool=False):
    """_summary_

    Args:
        problem (dict): The problem description. Should look like:
            dict{"num_vars": int, "names": list[str], "bounds": list[list]}
            where "names" has all names of the variables and "bounds" contains
            the bounds of these variables.
        from_file (bool, optional): Whether to take the data from a file.
            Defaults to True.
        save_as (str, optional): What to save the images as. Defaults to None.
        data (pd.DataFrame, optional): The data used for analysis. Defaults to
            None.
        second_order (bool, optional): Whether the second order sensitivity is
            analysed. Defaults to False.
    """
    if from_file:
        dfs = []
        # find all runs and concatenate dataframes
        path = get_output_path()
        files = os.listdir(f"{path}/run_data")
        for file in files:
            if file.startswith(save_as) and file.endswith(".csv"):
                # get data
                df = pd.read_csv(f"{path}/{file}")
                dfs.append(df)

        data = pd.concat(dfs)
        print(data.head())
        print(data.size)

    Si_polarization = sobol_analyze.analyze(problem, data["polarization"].values, calc_second_order=second_order, print_to_console=True)
    # Si_network_influence = sobol_analyze.analyze(problem, pd.concat(dfs[0:7])["network_influence"].values, calc_second_order=second_order, print_to_console=True)

    path = get_output_path()

    # first order
    plt.figure(dpi=300)
    plot_index(Si_polarization, problem['names'], '1', 'First order sensitivity')
    plt.savefig(f"{path}/{save_as}_polarization_1.png")
    plt.show()

    # second order
    if second_order:
        plt.figure(dpi=300)
        plot_index(Si_polarization, problem['names'], '2', 'Second order sensitivity')
        # plt.yscale()
        # plt.tight_layout()
        plt.tick_params(axis="y", labelsize=4)
        plt.tight_layout()
        plt.savefig(f"{path}/{save_as}_polarization_2.png")
        plt.show()

    # total order
    plt.figure(dpi=300)
    plot_index(Si_polarization, problem['names'], 'T', 'Total order sensitivity')
    plt.savefig(f"{path}/{save_as}_polarization_T.png")
    plt.show()


if __name__ == "__main__":

    save_as = "sensitivity_analysis_final"

    problem = {
    "num_vars": 12,
    "names": ["width",
                "lambd",
                "mu",
                "d1",
                "d2",
                "satisfaction_distance",
                "satisfaction_threshold",
                "network_type",
                "grid_preference",
                "grid_radius",
                "grid_density",
                "both_affected"],
    "bounds": [[5, 21],
                [0, 0.5],
                [0, 0.5],
                [0, np.sqrt(2)/2],
                [np.sqrt(2)/2, np.sqrt(2)],
                [0, np.sqrt(2)],
                [-0.5, 0.5],
                [0, len(Political_spectrum.network_types)],
                [0, 1],
                [1, 4],
                [0.5, 1],
                [0, 1]]}

    second_order = True

    samples = create_samples(problem=problem,
                            num_samples=64,
                            second_order=second_order,
                            save_data=True,
                            save_as=save_as)

    data = sobol_run_samples(problem=problem,
                            repeats=8,
                            max_steps=100,
                            data_collection_period=-1,
                            from_data=True,
                            save_as=save_as,
                            samples=None,
                            number_processes=None,
                            save_data=True)

    sobol_analyze_data(problem=problem,
                    from_file=True,
                    save_as=save_as,
                    second_order=second_order)
