from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import NetworkModule, CanvasGrid

from model import Political_spectrum

def color(agent) -> str:
    if agent.beliefs[0] < 0.5 and agent.beliefs[1] < 0.5:
        return "red"
    elif agent.beliefs[0] < 0.5:
        return "green"
    elif agent.beliefs[1] < 0.5:
        return "blue"
    else:
        # assert agent.beliefs[0] >= 0.5 and agent.beliefs[1] >= 0.5
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
                           "tooltip": "id: {}<br>state: {}".format(agents[0].unique_id, agents[0].beliefs[0]),
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
        "Color": color(agent),
        "Filled": "true",
        "Layer" : 0,
        "w": 0.9,
        "h": 0.9
    }
    return portrayal

width = 10
height = 10

grid = CanvasGrid(grid_portrayal, width, height, 500, 500)

lambd=0.05
mu=0.20
d1=0.35
d2=1.5
mu_norm=0.5
sigma_norm=0.45

server = ModularServer(
    Political_spectrum, 
    [network, grid], 
    "Politics Model", 
    {
        "width": width,
        "height": height,
        "lambd": lambd,
        "mu": mu,
        "d1": d1,
        "d2": d2,
        "mu_norm": mu_norm,
        "sigma_norm": sigma_norm,
        "network_type": "idealised"
    }
)
# width, height, tau, r
server.port = 8521  # The default
server.launch()
