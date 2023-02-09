# Modelling political beliefs
Model created for the UvA course Agent Based Modelling
Created by Anton Andersen, Chang Lin, Seda den Boer and Eva Plas.

## The Model
This repository contains the model for simulating political beliefs based on local interactions and interactions through social media. Please message one of the authors if you would like to read our paper on this model! The slides of the presentation done at the ABM course are in the `docs` folder.

The model is contained in the `code/classes` folder. The code for the model can be found in `model.py` and the code for the agents can be found in `agents.py`.

## Requirements
- install the required libraries (with the correct versions) through the requirements.txt by running: `pip install -r requirements.txt` from the home directory.

## Running the model
To run any of the files, move to the code folder. Any output will be generated in the output_files folder.
- To run the sensitivity analysis on the model, the file sensitivity_analysis.py should be run.
- Any visualisations are contained in the visualisation folder.
    - Run visualize_politics.py for a visualization of the model.
    - Run visualize_beliefs.py for a visualization of the agents beliefs changing over time.

