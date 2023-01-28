from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from mesa.batchrunner import batch_run

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from model import Political_spectrum

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

def run_batch_model(parameters):
    max_steps, data_collection_period, variables = parameters
    batch = batch_run(model_cls=Political_spectrum,
                        parameters=variables, # will run each combination of parameters
                        number_processes=1, # not parallel
                        iterations=1,
                        data_collection_period=data_collection_period,
                        max_steps=max_steps,
                        display_progress=False)

    data = pd.DataFrame(batch)
    return batch

def sobol_sensitivity_analysis(number_processes = None):
    problem = {
        "num_vars": 2,
        "names": ["d1", "d2"],
        "bounds": [[0.01, 1.0], [1.0, 1.5]]}

    repeats = 2
    max_steps = 100
    num_samples = 8 # should be a power of 2
    data_collection_period = -1

    param_values = np.array(sobol_sample.sample(problem, num_samples, calc_second_order=False))
    # print(param_values)
    # print(param_values[:,0])
    parameters = {}
    parameters_list = []
    for run in range(len(param_values)):
        variables = {}
        for name, val in zip(problem["names"], param_values[run]):
            variables[name] = val
        parameters[run] = variables
        parameters_list.append([max_steps, data_collection_period, variables])
    # for i, var in enumerate(problem["names"]):
    #     parameters[var] = list(param_values[:,i])

    # print(parameters)

    # create a dataframe to save the data
    data = pd.DataFrame(index=range(repeats*len(param_values)),
                        columns=problem["names"])
    data["polarization"] = None
    run = 0 # keep track of the runs for the dataframe
    print(data)


    if not number_processes:
        number_processes = mp.cpu_count()

    if number_processes > 1:
        pool = mp.Pool(number_processes)
        for rep in range(repeats):
            process = pool.map_async(run_batch_model, parameters_list)
            result = process.get()
            for i in range(len(result)):
                run_dict = result[i][0]
                print(run_dict)
                for name in data.columns:
                    data.loc[run, name] = run_dict[name]
                run += 1

        print(data)
        pool.close()

    else:
        for rep in range(repeats):
            for run in parameters:
                variables = parameters[run]
                batch = batch_run(model_cls=Political_spectrum,
                                parameters=variables, # will run each combination of parameters
                                number_processes=1, # not parallel
                                iterations=repeats,
                                data_collection_period=data_collection_period,
                                max_steps=max_steps,
                                display_progress=False)

                data = pd.DataFrame(batch)
                print(data.head())
                print(len(data))
        
    
    return data

if __name__ == "__main__":
    # ofat_sensitivity_analysis(False)
    sobol_sensitivity_analysis(number_processes=None)