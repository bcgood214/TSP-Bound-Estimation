import networkx as nx
import matplotlib.pyplot as plt\
  
# path = './testWeighted.dot'
path = './testWeighted2.graphml'
# path = './harary.graphml'

# graph = nx.Graph(nx.nx_agraph.read_dot(path))
graph = nx.read_graphml(path)


print(nx.display(graph))
print(nx.is_weighted(graph))
print(nx.get_edge_attributes(graph, 'color'))
print(graph.edges())
# nx.draw(graph)
# plt.draw()