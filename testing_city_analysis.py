from grammars import grammars
from cityGenerator import CityGenerator
import metrics_testing as metrics
# import generateMaps as maps
import numpy as np
from collections import defaultdict



cities_per_grammar = 5
population_goal = 500000


grammar_strings = [
    (grammars.Organic, "organic"),
    (grammars.Line, "line"),
    (grammars.Hex, "hex"),
    (grammars.Grid, "grid"),
]

results = defaultdict(lambda: {
    "roads_tau": [],
    "edges_tau": [],
    "score_metric_raw": [],
    "score_metric_norm_01": [],
    "score_metric_norm_04": [],
    "score_metric_norm_07": [],
})

for seed in range(cities_per_grammar):
    for grammar, grammar_str in grammar_strings:

        cityGen = CityGenerator()

        city = cityGen.generateCity(
            iterations=10000,
            population=population_goal,
            grammar=grammar,
            seed=seed,
            intersectRadius=0.8,
            plotType=None,
            verbose=True
        )

        shortestPaths, _ = metrics.calculateShortestPaths(city)

        rawEdgeBetweenness = metrics.calculateRawEdgeBetweenness(shortestPaths)

        rawRoadBetweenness = metrics.calculateRawRoadBetweenness(
            city, rawEdgeBetweenness
        )

        roads_tau = metrics.calculateLivingMetricKendallRoads(
            city, shortestPaths, rawRoadBetweenness
        )

        edges_tau = metrics.calculateLivingMetricKendallEdges(
            shortestPaths, rawEdgeBetweenness
        )

        score_metric_raw = metrics.calculateLivingMetricScore(
            city, rawRoadBetweenness
        )

        score_metric_normalised_01 = metrics.calculateLivingMetricScoreNormalised(
            city, rawRoadBetweenness, lambda_=0.1
        )

        score_metric_normalised_04 = metrics.calculateLivingMetricScoreNormalised(
            city, rawRoadBetweenness, lambda_=0.4
        )

        score_metric_normalised_07 = metrics.calculateLivingMetricScoreNormalised(
            city, rawRoadBetweenness, lambda_=0.7
        )

        # store results
        results[grammar_str]["roads_tau"].append(roads_tau)
        results[grammar_str]["edges_tau"].append(edges_tau)
        results[grammar_str]["score_metric_raw"].append(score_metric_raw)
        results[grammar_str]["score_metric_norm_01"].append(score_metric_normalised_01)
        results[grammar_str]["score_metric_norm_04"].append(score_metric_normalised_04)
        results[grammar_str]["score_metric_norm_07"].append(score_metric_normalised_07)


print("\n===== Living Metric Summary by Grammar =====\n")

for grammar_str, metrics_dict in results.items():

    print(f"Grammar: {grammar_str}")

    for metric_name, values in metrics_dict.items():
        values = np.array(values)

        mean = np.nanmean(values)
        std = np.nanstd(values)
        min_v = np.nanmin(values)
        max_v = np.nanmax(values)

        print(
            f"  {metric_name:20s} "
            f"mean = {mean: .3f}, "
            f"std = {std: .3f}, "
            f"min = {min_v: .3f}, "
            f"max = {max_v: .3f}"
        )

    print()


        # cityGen.saveCity(
        #     city,
        #     f"cities/{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        # )


        #shortestPathsSavePath = f"metadata/sp_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        #betweennessSavePath = f"metadata/centrality_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        

        #

