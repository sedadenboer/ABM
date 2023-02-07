import matplotlib.pyplot as plt
import networkx as nx

num_agents=40
A = nx.erdos_renyi_graph(num_agents, 0.4)
B = nx.barabasi_albert_graph(num_agents, 2)
C = nx.random_geometric_graph(num_agents, 0.15)
D = nx.complete_graph(num_agents)

plt.figure()
nx.draw(A, node_size=50)
plt.savefig("../normala.png")

plt.figure()
nx.draw(B, node_size=50)
plt.savefig("../normalb.png")

plt.figure()
nx.draw(C, node_size=50)
plt.savefig("../normalc.png")

plt.figure()
nx.draw(D, node_size=50)
plt.savefig("../normald.png")


# nx.draw_kamada_kawai(G, node_size=50)
# plt.savefig("../kamada_kawai.png")

# nx.draw_planar(G)
# plt.savefig("../planar.png")

# nx.draw_shell(G, node_size=50)
# plt.savefig("../shell.png")

# nx.draw_spring(G, node_size=50)
# plt.savefig("../spring.png")

# nx.draw_spectral(G, node_size=50)
# plt.savefig("../spectral.png")