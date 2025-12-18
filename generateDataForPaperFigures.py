from grammars import grammars
from cityGenerator import CityGenerator
import metrics
import generateMaps as maps
import pandas as pd

cities_per_grammar = 1
population_goal = 500000

grammar_strings = [
    (grammars.Organic, "organic"),
    (grammars.Line, "line"),
    (grammars.Hex, "hex"),
    (grammars.Grid, "grid"),
]

empirical_cities = {
        "london": ["London", 51.5072, -0.1275],
        "paris": ["Paris", 48.8567, 2.3522],
        "tokyo": ["Tokyo", 35.687, 139.7495],   # city, lat, lon
        "manhattan": ["Manhattan", 40.7834, -73.9662],
        "barcelona": ["Barcelona", 10.1403, -64.6833],
        "mexico_city": ["Mexico City", 19.4333, -99.1333],
}

# Create list of all grammars and seed
grammars_and_seeds = []
for empirical_city in empirical_cities.keys():
    for seed in range(3):
        grammars_and_seeds.append(["empirical_"+empirical_city, seed])
for seed in range(8):
    for grammar, grammar_str in grammar_strings:
        grammars_and_seeds.append([grammar_str, seed])

def add_row_to_file(file, row):
    with open(file, "a") as f:
        f.write(row+"\n")

# Generate metrics
data = []
for item in grammars_and_seeds:
    grammar_str, seed = item
    networkFile = f"cities/{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
    shortestPathsLoadPath = f"cities/shortestPaths_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
    betweennessLoadPath = f"cities/betweenness_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"

    # Load city
    cityGen = CityGenerator()
    city1 = cityGen.loadCity(networkFile)

    roadDensity = metrics.calculateRoadDensity(city1)
    meanLivingMetric = metrics.calcMeanLivingMetric(city1, par=5, betweennessLoadPath=betweennessLoadPath)
    meanCircuity = metrics.calculateAverageCircuity(city1, shortestPathsLoadPath)

    add_row_to_file("mean_circuity_R03.csv", f"{grammar_str},{500000},{seed},{meanCircuity}")
    add_row_to_file("road_density_R03.csv", f"{grammar_str},{500000},{seed},{roadDensity/10}")
    add_row_to_file("living_metric_beta_5_R03.csv", f"{grammar_str},{500000},{seed},{meanLivingMetric}")
