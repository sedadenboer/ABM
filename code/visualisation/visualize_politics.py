import mesa

from model import Political_spectrum

def color(agent) -> str:
    if agent.beliefs[0] >= 0.4 and agent.beliefs[0] < 0.6 and agent.beliefs[1] >= 0.4 and agent.beliefs[1] < 0.6:
        return "gray"
    elif agent.beliefs[0] < 0.5 and agent.beliefs[1] < 0.5:
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
        # print(G.nodes[source]['agent'][0])
        return G.nodes[source]['agent'][0], G.nodes[target]['agent'][0]

    # print(G.nodes.data("agent"))
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
    # print(portrayal)
    return portrayal

network = mesa.visualization.NetworkModule(network_portrayal)

def grid_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Color": color(agent),
        "Filled": "true",
        "Layer" : 0,
        # "w": 0.9,
        # "h": 0.9
        "r": 0.5
    }
    return portrayal

width = 10

grid = mesa.visualization.CanvasGrid(grid_portrayal, width, width)

lambd=0.5
mu=0.20
d1=0.7
d2=1.0

server = mesa.visualization.ModularServer(
    Political_spectrum,
    # [network, grid],
    [network],
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

# server.port = 8521  # The default
server.launch()
