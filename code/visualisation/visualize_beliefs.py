# visualize_beliefs.py
#
# Course: ABM (2022/2023)
#
# Description: File to create plots to visualize the
# opinion distribution of the agents.

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from classes.model import Political_spectrum
import time
from save_results.output import get_output_path
timestr = time.strftime("%Y%m%d-%H%M%S")
output_path = get_output_path()


def get_beliefs(model: Political_spectrum):
    """
    Retrieves all of the beliefs from the agents in the model.

    Args:
        model: political spectrum model

    Returns: both types of beliefs
    """
    x = []
    y = []
    for agent_id in model.agents:
        agent = model.agents[agent_id]
        belief_x, belief_y = agent.beliefs
        x.append(belief_x)
        y.append(belief_y)

    return x, y

def plot_beliefs(model: Political_spectrum):
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

    plt.savefig(f"{output_path}/{timestr}_scatterplot_step{num_steps}.png")

def animate_beliefs(model: Political_spectrum):
    """
    Make an animation of the beliefs over the course of time.

    Args:
        model: political spectrum model
    """
    assert model.num_steps == 0

    x, y = get_beliefs(model)

    # make animation
    fig, ax = plt.subplots(dpi=300)
    colors = [i / 100 for i in range(len(x))]
    scat = ax.scatter(x, y, c=colors, cmap="hsv")

    def animate(i):
        model.step()
        x, y = get_beliefs(model)
        points = [(x[j], y[j]) for j in range(len(x))]
        scat.set_offsets(points)

    ani = animation.FuncAnimation(fig, animate, frames=10, interval=10)
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)

    ani.save(f"{output_path}/{timestr}_scatter.gif", writer=writer)
    

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

    plot_beliefs(model)
    animate_beliefs(model)