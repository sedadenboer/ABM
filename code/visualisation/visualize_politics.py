# visualize_politics.py
#
# Course: ABM (2022/2023)
#
# Description: File to run mesa opinion visualisation of the model.
# This includes the grid and the network

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mesa
from classes.model import Political_spectrum


def color(agent):
    if agent.beliefs[0] >= 0.4 and agent.beliefs[0] < 0.6 and agent.beliefs[1] >= 0.4 and agent.beliefs[1] < 0.6:
        return "gray"
    elif agent.beliefs[0] < 0.5 and agent.beliefs[1] < 0.5:
        return "red"
    elif agent.beliefs[0] < 0.5:
        return "green"
    elif agent.beliefs[1] < 0.5:
        return "blue"
    else:
        return "yellow"

def network_portrayal(G):
    def node_color(agent):
        return color(agent)
    
    def edge_color(agent1, agent2):
        return "black"
    
    def edge_width(agent1, agent2):
        return 2
    
    def get_agents(source, target):
        return G.nodes[source]['agent'][0], G.nodes[target]['agent'][0]

    portrayal = dict()
    portrayal["nodes"] = [{"size": 6,
                           "color": node_color(agents[0]),
                           "tooltip": "id: {}<br>state: {}".format(unique_id, agents[0].beliefs[0]),
                           }
                          for (unique_id, agents) in G.nodes.data("agent")]
    portrayal['edges'] = [{'source': source,
                           'target': target,
                           'color': edge_color(*get_agents(source, target)),
                           'width': edge_width(*get_agents(source, target)),
                           }
                          for (source, target) in G.edges]

    return portrayal

def grid_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Color": color(agent),
        "Filled": "true",
        "Layer" : 0,
        "r": 0.5
    }

    return portrayal



if __name__ == "__main__":
    width = 10
    lambd=0.5
    mu=0.20
    d1=0.7
    d2=1.0

    grid = mesa.visualization.CanvasGrid(grid_portrayal, width, width)
    network = mesa.visualization.NetworkModule(network_portrayal)

    server = mesa.visualization.ModularServer(
        Political_spectrum,
        [network, grid],
        "Politics Model", 
        {
            "width": width,
            "lambd": lambd,
            "mu": mu,
            "d1": d1,
            "d2": d2,
            "network_type": 3,
            "grid_density": 0.95
        }
    )

    server.port = 8521  # The default
    server.launch()
