from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from mesa.batchrunner import batch_run

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp
import os

from itertools import combinations

from model import Political_spectrum

from visualize_beliefs import get_output_path

# OFAT
def ofat_sensitivity_analysis(from_data: bool, save_data=True):

    problem = {
        "num_vars": 1,
        "names": ["d1"],
        "bounds": [[0.01, 1.0]]}
    
    repeats = 10
    max_steps = 100
    samples_between_bounds = 30
    data_collection_period = -1 # collect data at the end

    data = {}

    for i, var in enumerate(problem["names"]):
        # create the samples for this variable
        samples = np.linspace(*problem["bounds"][i], num=samples_between_bounds) # these are floats

        # for _ in range(repeats):
        batch = batch_run(model_cls=Political_spectrum,
                        parameters={var: samples},
                        number_processes=None, # this makes it parallel
                        data_collection_period=data_collection_period,
                        iterations=repeats,
                        max_steps=max_steps,
                        display_progress=False)

        # print(batch) # this is a list with dicts:
        # start of runid 29
        # {'RunId': 29, 'iteration': 0, 'Step': 100, 'd1': 1.0, 'polarization': None, 'step': 100, 'AgentID': 0, 'beliefs': array([0., 1.])}
        # end of runid 29
        # {'RunId': 29, 'iteration': 0, 'Step': 100, 'd1': 1.0, 'polarization': None, 'step': 100, 'AgentID': 99, 'beliefs': array([1., 0.])}
        data = pd.DataFrame(batch)
        print(data.head())

    # all it needs is pretty plots!

def create_samples(problem, num_samples, second_order: bool, save_data: bool = False, save_as: str = None):
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
        variables["grid_radius"] = int(variables["grid_radius"])
        variables["both_affected"] = int(variables["both_affected"] > 0.5)

        parameters[run] = variables
        parameters_list.append(variables)

    if save_data:
        df = pd.DataFrame(parameters_list)
        path = get_output_path()
        df.to_csv(f"{path}/samples/{save_as}_samples.csv", index=False)
        
    return parameters_list


def run_batch_model(parameters):
    max_steps, data_collection_period, variables = parameters
    batch = batch_run(model_cls=Political_spectrum,
                        parameters=variables, # will run each combination of parameters
                        number_processes=1, # not parallel
                        iterations=1,
                        data_collection_period=data_collection_period,
                        max_steps=max_steps,
                        display_progress=False)
    return batch


def sobol_run_samples(problem, repeats, max_steps, data_collection_period, from_data: bool, save_as: str=None, samples=None, number_processes = None, save_data: bool = False):

    if from_data:
        # load the samples
        path = get_output_path()
        df = pd.read_csv(f"{path}/samples/{save_as}_samples.csv")
        samples = df.to_dict("records")
    else:
        # make sure samples are given
        if not samples:
            raise ValueError("Please provide samples if not run from a file.")

    parameters = [[max_steps, data_collection_period, i] for i in samples]

    # use the number of processors of this machine
    if not number_processes:
        number_processes = mp.cpu_count()

    if number_processes > 1:
        pool = mp.Pool(number_processes)

        for rep in range(repeats):
            # create a dataframe to save the data
            data = pd.DataFrame(index=range(len(samples)),
                                columns=problem["names"])
            data["polarization"] = None
            data["network_influence"] = None
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
                data.to_csv(f"{path}/run_data/{save_as}_{user}_{rep}.csv", index=False)

            print(data)
        pool.close()

    else:
        raise NotImplementedError()

    return data

def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
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

def sobol_analyze_data(problem, from_file: bool=True, save_as: str=None, data=None, second_order:bool=False):
    if from_file:
        dfs = []
        # find all runs and concatenate dataframes
        path = get_output_path()
        files = os.listdir(f"{path}/run_data")
        for file in files:
            if file.startswith(save_as) and file.endswith(".csv"):
                # get data
                df = pd.read_csv(f"{path}/run_data/{file}")
                dfs.append(df)

        data = pd.concat(dfs)
        print(data.head())
        print(data.size)

    Si_polarization = sobol_analyze.analyze(problem, data["polarization"].values, calc_second_order=second_order, print_to_console=True)
    Si_network_influence = sobol_analyze.analyze(problem, pd.concat(dfs[0:7])["network_influence"].values, calc_second_order=second_order, print_to_console=True)

    path = get_output_path()

    # first order
    plot_index(Si_polarization, problem['names'], '1', 'First order sensitivity')
    plt.savefig(f"{path}/images/{save_as}_polarization_1.png")
    plt.show()

    # second order
    if second_order:
        plot_index(Si_polarization, problem['names'], '2', 'Second order sensitivity')
        plt.savefig(f"{path}/images/{save_as}_polarization_2.png")
        plt.show()

    # total order
    plot_index(Si_polarization, problem['names'], 'T', 'Total order sensitivity')
    plt.savefig(f"{path}/images/{save_as}_polarization_T.png")
    plt.show()


    network_names = problem['names'].copy()
    network_names.remove("network_type")
    network_names.remove("mu")
    print(network_names)
    copy_Si = Si_network_influence.copy()
    for key in copy_Si:
        value = copy_Si[key]
        print(value)
        value = np.concatenate([value[:1], value[2:4], value[5:]])
        copy_Si[key] = value
        print(value)
    # first order
    plot_index(copy_Si, network_names, '1', 'First order sensitivity')
    plt.savefig(f"{path}/images/{save_as}_networkinfluence_1.png")
    plt.show()

    if second_order:
        plot_index(copy_Si, network_names, '2', 'Second order sensitivity')
        plt.savefig(f"{path}/images/{save_as}_networkinfluence_2.png")
        plt.show()

    # total order
    plot_index(copy_Si, network_names, 'T', 'Total order sensitivity')
    plt.savefig(f"{path}/images/{save_as}_networkinfluence_T.png")
    plt.show()

if __name__ == "__main__":

    save_as = "new_sa"

    # problem = {
    # "num_vars": 8,
    # "names": ["lambd",
    #             "mu",
    #             "d1",
    #             "d2",
    #             "network_type",
    #             "grid_preference",
    #             "grid_radius",
    #             "both_affected"],
    # "bounds": [[0, 0.5],
    #             [0, 0.5],
    #             [0, np.sqrt(2)/2],
    #             [np.sqrt(2)/2, np.sqrt(2)],
    #             [0, len(Political_spectrum.network_types)],
    #             [0, 1],
    #             [1, 4],
    #             [0,1]]
    # }

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
    "bounds": [[5, 33],
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

    second_order = False

    samples = create_samples(problem=problem,
                            num_samples=64,
                            second_order=second_order,
                            save_data=True,
                            save_as=save_as)

    # data = sobol_run_samples(problem=problem,
    #                         repeats=8,
    #                         max_steps=100,
    #                         data_collection_period=-1,
    #                         from_data=True,
    #                         save_as=save_as,
    #                         samples=None,
    #                         number_processes=None,
    #                         save_data=True)

    # sobol_analyze_data(problem=problem,
    #                 from_file=True,
    #                 save_as=save_as,
    #                 second_order=second_order)
