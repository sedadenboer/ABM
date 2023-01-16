from mesa import Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from math import sin, cos, pi
from agents import Individual

class Political_model(Model):
    def __init__(self, width, height, tau, r, radius: int = 1) -> None:
        self.num_agents = width * height
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        # create agents for each position in the grid
        self.num_A = 0
        self.num_B = 0
        self.create_agents()

        self.datacollector = DataCollector()

        self.alpha_AB = 1 + r * sin(tau)
        self.alpha_BA = 1 + r * cos(tau)
        print(self.alpha_AB, self.alpha_BA)
        self.radius = radius

        self.running = True
    
    def create_agents(self):
        unique_id = 0
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                pos = (x, y)
                if self.random.random() > 0.5:
                    opinion = "A"
                    self.num_A += 1
                else:
                    opinion = "B"
                    self.num_B += 1
                agent = Individual(unique_id, self, pos, opinion)
                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)
                unique_id += 1

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self.num_A == 0 or self.num_B == 0:
            self.running = False

    def run(self, iterations):
        for _ in range(iterations):
            self.step()

if __name__ == "__main__":
    model = Political_model(3, 15, pi/2, 0.5)
    model.run(10)