import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from model import Political_spectrum

def get_beliefs(model: Political_spectrum):
    x = []
    y = []
    for agent_id in model.agents:
        agent = model.agents[agent_id]
        belief_x, belief_y = agent.beliefs
        x.append(belief_x)
        y.append(belief_y)
    return x, y

def plot_beliefs(model: Political_spectrum, run_id: int):
    """Plots the current state of beliefs of the agent in the model.

    Args:
        model (Political_spectrum): The model for which the beliefs will be modelled.
    """
    x, y = get_beliefs(model)

    plt.figure(dpi=300)
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.plot(x, y, ".")
    num_steps = model.num_steps
    plt.savefig(f"../output_files/{run_id}_scatterplot_step{num_steps}.png")

def animate_beliefs(model: Political_spectrum, run_id: int):
    assert model.num_steps == 0

    x, y = get_beliefs(model)

    # make animation
    fig, ax = plt.subplots(dpi=300)
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.0])

    # points = [(x[i], y[i]) for i in range(len(x))]
    colors = [i / 100 for i in range(len(x))]
    cmap = "hsv"
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
    ani.save(f'../output_files/{run_id}_scatter.gif', writer=writer)

if __name__ == "__main__":
    width = 10
    height = 10
    lambd=0.05
    mu=0.05
    d1=0.5
    d2=1.5
    mu_norm=0.5
    sigma_norm=0.2
    network_type = "BA"
    grid_preference = 0.5

    model = Political_spectrum(width, height, lambd, mu, d1, d2, mu_norm, sigma_norm, network_type, grid_preference)
    run_id = datetime.datetime.now()
    plot_beliefs(model, run_id)
    # for _ in range(5):
    #     for _ in range(100):
    #         model.step()
    #     plot_beliefs(model, run_id)

    animate_beliefs(model, run_id)



