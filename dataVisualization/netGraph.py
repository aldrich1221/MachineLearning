import networkx as nx 
import matplotlib.pyplot as plt

# G = nx.Graph()
# G = nx.DiGraph()
G = nx.MultiGraph()
#G = nx.MultiDiGraph()

# G.add_node('a')
# #G.add_node(1,1)
# G.add_edge('x','y')
# G.add_weight_edges_from([('x','y',1.0)])

# list = [[('a','b',5.0),('b','c',3.0),('a','c',1.0)]
# #G.add_weight_edges_from([(list)])
# nx.draw(G)

# plt.show()

G.add_nodes_from(['b', 'c', 'd', 'e'])
G.add_cycle(['f', 'g', 'h', 'j'])

H = nx.path_graph(10)


G.add_nodes_from(H)
G.add_node(H)


#G = nx.random_graphs.barabasi_albert_graph(100,1)
nx.draw(G,with_labels=True,node_color='red')
plt.show()