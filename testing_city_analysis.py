from grammars import grammars
from cityGenerator import CityGenerator
import metrics_testing as metrics
# import generateMaps as maps



cities_per_grammar = 1
population_goal = 500000


grammar_strings = [
    (grammars.Organic, "organic"),
    (grammars.Line, "line"),
    (grammars.Hex, "hex"),
    (grammars.Grid, "grid"),
]


for seed in range(cities_per_grammar):
    for grammar, grammar_str in grammar_strings:

        cityGen = CityGenerator()

        city = cityGen.generateCity(
            iterations=10000,
            population=population_goal,
            grammar=grammar,
            seed=seed,
            intersectRadius=0.8,
            plotType="Map"
        )


        shortestPaths, _ = metrics.calculateShortestPaths(city)

        rawEdgeBetweenness = metrics.calculateRawEdgeBetweenness(shortestPaths)

        rawRoadBetweenness = metrics.calculateRawRoadBetweenness(city, rawEdgeBetweenness)

        mean_tau = metrics.calculateLivingMetricV2(city, shortestPaths, rawRoadBetweenness)

        print(f"{grammar_str}: {mean_tau}")


        # cityGen.saveCity(
        #     city,
        #     f"cities/{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        # )


        #shortestPathsSavePath = f"metadata/sp_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        #betweennessSavePath = f"metadata/centrality_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        

        #

