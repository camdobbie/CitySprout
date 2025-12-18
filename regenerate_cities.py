from grammars import grammars
from cityGenerator import CityGenerator
import metrics
import generateMaps as maps
import osmnx as ox
import numpy as np
import networkx
import matplotlib.pyplot as plt

cities_per_grammar = 100
population_goal = 500000
generate_synthetic_cities = True
generate_empirical_cities = False

grammar_strings = [
    (grammars.Organic, "organic"),
    (grammars.Line, "line"),
    (grammars.Hex, "hex"),
    (grammars.Grid, "grid"),
]

## Empirical cities
def amend_empirical_city(G):
    G_proj = ox.project_graph(G)
    # Add 'pos' attribute
    for index, node in G_proj.nodes.items():
        node["pos"] = np.array([node["x"], node["y"]])
    # Add road numbers
    next_road_number = 0
    road_numbers = {}
    unique_index = 0
    for edge_index, edge in G_proj.edges.items():
        unique_index += 1
        try:
            key = edge["name"]
        except Exception as e:
            key = unique_index
        if isinstance(key, list):
            key = key[0]
        if not key in road_numbers.keys():
            road_numbers[key] = next_road_number
            next_road_number += 1
        edge['roadNumber'] = road_numbers[key]
    # Return
    return networkx.classes.multidigraph.DiGraph(G_proj)

def get_city_from_point(lon, lat, radius=10000, network_type="drive"):
    print("Load...")
    try:
        G = ox.graph.graph_from_point(center_point=(lat,lon), dist=radius, dist_type='bbox', network_type=network_type, simplify=True, retain_all=False, truncate_by_edge=False, custom_filter=None)
    except Exception as e:
        print(e)
    print("Amend...")
    return amend_empirical_city(G)

def get_city_by_name(name, network_type="drive"):
    print("Download", name, "...")
    G = ox.graph_from_place(
        name,
        network_type=network_type,   # drive=roads usable by cars
        simplify=True,
    )
    return amend_empirical_city(G)

if generate_empirical_cities:
    cities = {
        # city, lat, lon
        "london": ["London", 51.5072, -0.1275],
        "paris": ["Paris", 48.8567, 2.3522],
        #"tokyo": ["Tokyo", 35.687, 139.7495],
        #"manhattan": ["Manhattan", 40.7834, -73.9662],
        #"barcelona": ["Barcelona", 10.1403, -64.6833],
        #"dubai": ["Dubai", 25.2631, 55.2972],
        #"brasilia": ["Brasilia", -15.7939, -47.8828],
        #"mexico_city": ["Mexico City", 19.4333, -99.1333],
    }
    for city_name_short, city_data in cities.items():
        for seed in [0, 1, 2]:
            city_name, lat, lon = city_data
            print("Load", city_name)
            #city = get_city_from_point(lon, lat, radius=400, network_type="all")
            if seed==1:
                lon += 0.05
            elif seed==2:
                lat += 0.05
            city = get_city_from_point(lon, lat, radius=1000, network_type="drive")
            #city = get_city_by_name(city_name, network_type="drive")

            # Draw city
            if seed==0:
                plt.clf() 
                pos = networkx.get_node_attributes(city, 'pos')
                networkx.draw_networkx_edges(city, pos, edge_color='black', width=1, arrows=False)
                plt.savefig(city_name_short+".png")

            cityGen = CityGenerator()
            cityGen.saveCity(
                city,
                f"cities/empirical_{city_name_short}_seed_{seed}_pop_{population_goal}.pkl"
            )
            # Calc shortest path and betweenness centrality
            shortestPathsSavePath = f"cities/shortestPaths_empirical_{city_name_short}_seed_{seed}_pop_{population_goal}.pkl"
            betweennessSavePath = f"cities/betweenness_empirical_{city_name_short}_seed_{seed}_pop_{population_goal}.pkl"
            metrics.calculateRoadBetweennessCentrality(city, shortestPathsSavePath, betweennessSavePath)


## Generated cities
if generate_synthetic_cities:
    for seed in range(cities_per_grammar):
        for grammar, grammar_str in grammar_strings:
            cityGen = CityGenerator()
            city = cityGen.generateCity(
                iterations=10000,
                population=population_goal,
                grammar=grammar,
                seed=seed,
                intersectRadius=0.8,
                plotType="none"
            )
            cityGen.saveCity(
                city,
                f"cities/{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
            )
            # Calc shortest path and betweenness centrality
            shortestPathsSavePath = f"cities/shortestPaths_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
            betweennessSavePath = f"cities/betweenness_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
            metrics.calculateRoadBetweennessCentrality(city, shortestPathsSavePath, betweennessSavePath)
