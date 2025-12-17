from grammars import grammars
from cityGenerator import CityGenerator
import metrics
import generateMaps as maps



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
        cityGen.saveCity(
            city,
            f"cities/{grammar_str}_seed_{seed}_pop_{population_goal}.pkl"
        )

