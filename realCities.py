import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

class EmpiricalCities:
    def __init__(self):
        self.cities = {
            "Royston": "Royston, UK",
            "Milton Keynes": "Milton Keynes, UK",
            "London": "London, UK",
            "Barcelona": "Barcelona, Spain",
            "New York": "New York City, USA",
            "Paris": "Paris, France",
        }
        self.graphs = {}
        for name, query in self.cities.items():
            print("Download", name, "...")
            G = ox.graph_from_place(
                query,
                network_type="drive",   # roads usable by cars
                simplify=True
            )
            print("Done!")
            self.graphs[name] = G

    def plot(self, city_name):
        ox.plot_graph(
            self.graphs[city_name],
            node_size=0,
            edge_linewidth=0.6
        )

    def get_metric_road_density(self, city_name):
        G = self.graphs[city_name]
        # Project network to metres x, y
        G_proj = ox.project_graph(G)
        # Process all links
        x_min, y_min, x_max, y_max = None, None, None, None
        total_road_length = 0
        for edge in list(G_proj.edges(keys=True, data=True)):
            u, v, k, data = edge
            x0, y0 = G_proj.nodes[u]["x"], G_proj.nodes[u]["y"]
            x1, y1 = G_proj.nodes[v]["x"], G_proj.nodes[v]["y"]
            if x_min is None or x0<x_min or x1<x_min:
                x_min = min(x0, x1)
            if y_min is None or y0<y_min or y1<y_min:
                y_min = min(y0, y1)
            if x_max is None or x0>x_max or x1>x_max:
                x_max = min(x0, x1)
            if y_max is None or y0>y_max or y1>y_max:
                y_max = min(y0, y1)
            total_road_length += data["length"]
        D = total_road_length / ((x_max-x_min)*(y_max-y_min))
        return D

    def get_circuity_metric(self, city_name):
        G = self.graphs[city_name]
        # Project network to metres x, y
        G_proj = ox.project_graph(G)
        # Process all edges
        for edge in list(G_proj.edges(keys=True, data=True)):
            u, v, k, data = edge
            unweighted_path = nx.shortest_path(G_proj, source=u, target=v)
        # Not sure how to do this!

    def get_metric_LM(self, city_name):
        G = self.graphs[city_name]
        # Calc betweenness centrality
        edge_bc = nx.edge_betweenness_centrality(
            G,
            weight="length",
            k=1000,          # number of sampled source nodes
            normalized=True,
            seed=42
        )
        # Attach results back to edges
        for (u, v, k), value in edge_bc.items():
            G.edges[u, v, k]["edge_bc"] = value
        # Rank all edges by BR
        # Not sure how to do this!


# Load all cities
ec = EmpiricalCities()

# # Calculate metrics
# for city_key, city_name in ec.cities.items():
#     D = m1 = ec.get_metric_road_density(city_key)
#     print(city_key, D)

for city_key, city_name in ec.cities.items():
    ec.plot(city_key)
    plt.show()
