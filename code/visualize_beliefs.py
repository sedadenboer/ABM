import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import os
from pathlib import Path
import numpy as np

from model import Political_spectrum

def get_output_path():
    path = os.path.abspath(__file__)
    output_path = Path(path).parent
    # move to the parent if still in the code directory
    if os.path.split(output_path)[1] == "code":
        output_path = output_path.parent
    output_path = output_path / "output_files"
    return output_path

def get_beliefs(model: Political_spectrum):
    x = []
    y = []
    for agent_id in model.agents:
        agent = model.agents[agent_id]
        belief_x, belief_y = agent.beliefs
        x.append(belief_x)
        y.append(belief_y)
    return x, y

def plot_beliefs(model: Political_spectrum, run_id):
    """Plots the current state of beliefs of the agent in the model.

    Args:
        model (Political_spectrum): The model for which the beliefs will be modelled.
    """
    x, y = get_beliefs(model)

    plt.figure(dpi=300)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.plot(x, y, ".")
    plt.plot([0.5, 0.5], [0.0, 1.0], "k", alpha=0.5, label="_not in legend")
    plt.plot([0.0, 1.0], [0.5, 0.5], "k", alpha=0.5, label="_not in legend")
    plt.title("Distribution of agents' beliefs")
    num_steps = model.num_steps

    output_path = get_output_path()
    plt.savefig(f"{output_path}/images/{run_id}_scatterplot_step{num_steps}.png")

def animate_beliefs(model: Political_spectrum, run_id: int):
    assert model.num_steps == 0

    x, y = get_beliefs(model)

    # make animation
    fig, ax = plt.subplots(dpi=300)
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.0])

    # points = [(x[i], y[i]) for i in range(len(x))]
    colors = [i / 100 for i in range(len(x))]
    scat = ax.scatter(x, y, c=colors, cmap="hsv")
    # scat = ax.scatter(points)

    def animate(i):
        model.step()
        x, y = get_beliefs(model)
        # print(x)
        points = [(x[j], y[j]) for j in range(len(x))]

        # scat.set_offsets((x, y))
        scat.set_offsets(points)

    ani = animation.FuncAnimation(fig, animate, frames=10, interval=10)

    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    path = get_output_path()
    ani.save(f'{path}/images/{run_id}_scatter.gif', writer=writer)

def gradient_beliefs(model: Political_spectrum):
    x, y = get_beliefs(model)
    print(x)
    print(y)
    print()

    Z = np.random.rand(3, 5)
    print()
    print(Z)
    plt.figure()
    plt.pcolor(Z)
    plt.show()
    

if __name__ == "__main__":
    model = Political_spectrum(width=20,
                                lambd=0.05,
                                mu=0.20,
                                d1=0.35,
                                d2=1.5,
                                mu_norm=0.5,
                                sigma_norm=0.2,
                                network_type="BA",
                                grid_preference=0.5)
    run_id = datetime.datetime.now()
    plot_beliefs(model, run_id)

    # for _ in range(5):
    #     for _ in range(100):
    #         model.step()
    #     plot_beliefs(model, run_id)

    # animate_beliefs(model, run_id)

    # gradient_beliefs(model)





