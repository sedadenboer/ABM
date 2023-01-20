import matplotlib.pyplot as plt
import networkx as nx

G = nx.barabasi_albert_graph(n=100, m=2)
nx.draw_kamada_kawai(G, node_size=50)
plt.savefig("../kamada_kawai.png")

# nx.draw_planar(G)
# plt.savefig("../planar.png")

nx.draw_shell(G, node_size=50)
plt.savefig("../shell.png")

nx.draw_spring(G, node_size=50)
plt.savefig("../spring.png")

nx.draw_spectral(G, node_size=50)
plt.savefig("../spectral.png")