from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import NetworkModule, CanvasGrid

from model import Political_spectrum

def network_portrayal(G):
    def node_color(agent):
        return "red" if agent.opinion < 0.5 else "blue"
    
    def edge_color(agent1, agent2):
        return "black"
    
    def edge_width(agent1, agent2):
        return 2
    
    def get_agents(source, target):
        return G.nodes[source]['agent'][0], G.nodes[target]['agent'][0]
    
    portrayal = dict()
    portrayal["nodes"] = [{"size": 6,
                           "color": node_color(agents[0]),
                           "tooltip": "id: {}<br>state: {}".format(agents[0].unique_id, agents[0].opinion),
                           }
                          for (_, agents) in G.nodes.data("agent")]
    portrayal['edges'] = [{'source': source,
                           'target': target,
                           'color': edge_color(*get_agents(source, target)),
                           'width': edge_width(*get_agents(source, target)),
                           }
                          for (source, target) in G.edges]

    return portrayal

network = NetworkModule(network_portrayal, 500, 500, library="d3")

def grid_portrayal(agent):
    portrayal = {
        "Shape": "rect",
        "Color": "red" if agent.opinion < 0.5 else "blue",
        "Filled": "true",
        "Layer" : 0,
        "w": 0.9,
        "h": 0.9
    }
    return portrayal

width = 3
height = 3

grid = CanvasGrid(grid_portrayal, width, height, 500, 500)

server = ModularServer(
    Political_spectrum, 
    [network, grid], 
    "Politics Model", 
    {"width": width, "height": height}
)
# width, height, tau, r
server.port = 8521  # The default
server.launch()