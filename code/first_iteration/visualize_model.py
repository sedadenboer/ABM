from mesa.visualization.modules.CanvasGridVisualization import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from model import Political_model
from math import pi

import argparse

parser = argparse.ArgumentParser(description = "Visualizes the politics model",)
parser.add_argument("tau", type=float)

args = parser.parse_args()

def agent_portrayal(agent):
    portrayal = {
        "Shape": "rect",
        "Color": "red" if agent.opinion == "A" else "blue",
        "Filled": "true",
        "Layer" : 0,
        "w": 0.9,
        "h": 0.9
    }
    return portrayal

width = 50
height = 50

grid = CanvasGrid(agent_portrayal, width, height, 700, 700)

server = ModularServer(
    Political_model, [grid], "Politics Model", {"width": width, "height": height, "tau": args.tau * pi, "r": 0.5}
)
# width, height, tau, r
server.port = 8521  # The default
server.launch()