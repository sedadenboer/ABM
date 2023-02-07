from model import Political_spectrum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from visualize_beliefs import get_output_path

def color(agent):
    # make a gradient
    pass

def visualize_network(model: Political_spectrum):
    network = model.network.G
    for node_id in network.nodes:
        agent = network.nodes[node_id]["agent"][0]
        print(agent)

def visualize_grid(model: Political_spectrum):
    grid = model.grid
    for x in range(grid.width):
        for y in range(grid.height):
            agent = grid[x][y]
            print(agent)

def visualize_beliefs(model: Political_spectrum):
    pass

def sample_beliefs():
    x = list(np.linspace(0, 1, 100))
    x_grid = np.array([x for _ in range(len(x))])
    # print(x_grid)
    y = x.copy()
    y_grid = np.array([[i]*len(y) for i in y])
    # print(y_grid)
    x_colormap = mpl.colormaps["bwr"]
    y_colormap = mpl.colormaps["PiYG"]
    
    fig, axs = plt.subplots(1, 2, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    axes = [ax for ax in axs.flat]
    psm = axes[0].pcolormesh(x_grid, cmap=x_colormap, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(psm, ax=axes[0])
    psm = axes[1].pcolormesh(y_grid, cmap=y_colormap, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(psm, ax=axes[0])
    plt.show()
    path = get_output_path()
    plt.savefig(f"{path}/images/AA.png")
    

if __name__ == "__main__":
    # model = Political_spectrum()
    # visualize_network(model)
    # visualize_grid(model)
    # visualize_beliefs(model)
    sample_beliefs()
