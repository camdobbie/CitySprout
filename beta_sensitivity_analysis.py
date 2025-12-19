from grammars import grammars
from cityGenerator import CityGenerator
import metrics
import pandas as pd
from generatePaperFigures import FigureGenerator

csv_file = "beta_analysis.csv"
generate_csv_file = False
generate_plot = True

## Generate plot
if generate_plot:
    fig_gen = FigureGenerator()
    fig_gen.plot_beta_sensitivity(save_path="beta_sensitivity_analysis.pdf", show=True, csv_file=csv_file)


## Generate CSV file
if generate_csv_file:
    cities_per_grammar = 30
    population_goal = 500000

    grammar_strings = [
        (grammars.Organic, "organic"),
        (grammars.Line, "line"),
        (grammars.Hex, "hex"),
        (grammars.Grid, "grid"),
    ]

    data = []
    for seed in range(cities_per_grammar):
        for grammar, grammar_str in grammar_strings:
            networkFile = f"cities/{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
            shortestPathsLoadPath = f"cities/shortestPaths_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
            betweennessLoadPath = f"cities/betweenness_{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"

            # Load city
            cityGen = CityGenerator()
            city1 = cityGen.loadCity(networkFile)

            for beta in range(1, 20, 1):
                meanLivingMetric = metrics.calcMeanLivingMetric(city1, par=beta, betweennessLoadPath=betweennessLoadPath)
                data.append([grammar_str.capitalize(), seed, beta, meanLivingMetric])

    df = pd.DataFrame(data)
    df.columns = ["grammar", "seed", "beta", "living_metric"]
    print(df)
    df.to_csv(csv_file)
