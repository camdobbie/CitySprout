from grammars import grammars
from cityGenerator import CityGenerator
import metrics_testing as metrics

import numpy as np
import csv
from collections import defaultdict



def generate_cities_and_save_scores():

    cities_per_grammar = 5
    population_goal = 500000

    lambdas = [0.1, 0.4, 0.7]

    grammar_strings = [
        (grammars.Organic, "organic"),
        (grammars.Line, "line"),
        (grammars.Hex, "hex"),
        (grammars.Grid, "grid"),
    ]


    csv_rows = []

    summary = defaultdict(lambda: {
        0.1: [],
        0.4: [],
        0.7: [],
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
                verbose=True,
            )

            shortestPaths, _ = metrics.calculateShortestPaths(city)
            rawRoadBetweenness = metrics.calculateRawRoadBetweenness(
                city,
                metrics.calculateRawEdgeBetweenness(shortestPaths),
            )

            for lambda_ in lambdas:
                score = metrics.calculateLivingMetricScoreNormalised(
                    city,
                    rawRoadBetweenness,
                    lambda_=lambda_,
                )

                csv_rows.append({
                    "grammar": grammar_str,
                    "seed": seed,
                    "lambda": lambda_,
                    "living_metric_score_value": score,
                })

                summary[grammar_str][lambda_].append(score)



    print("\n===== Normalised Living Metric Summary by Grammar =====\n")

    for grammar_str, lambda_dict in summary.items():
        print(f"Grammar: {grammar_str}")

        for lambda_, values in lambda_dict.items():
            values = np.array(values)

            mean = np.nanmean(values)
            std = np.nanstd(values)
            min_v = np.nanmin(values)
            max_v = np.nanmax(values)

            print(
                f"  lambda = {lambda_:>3.1f}  "
                f"mean = {mean: .3f}, "
                f"std = {std: .3f}, "
                f"min = {min_v: .3f}, "
                f"max = {max_v: .3f}"
            )

        print()



    csv_path = "living_metric_scores_normalised.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "grammar",
                "seed",
                "lambda",
                "living_metric_score_value",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved normalised living metric scores to {csv_path}")




def calculate_and_plot_correlation(
    v1_csv_path="living_metric_beta_5.csv",
    v2_csv_path="living_metric_scores_normalised.csv",
):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr



    df_v1 = pd.read_csv(
        v1_csv_path,
        header=None,
        names=["grammar", "population", "seed", "v1_score"],
    )

    df_v2 = pd.read_csv(v2_csv_path)

    # normalise grammar naming
    df_v1["grammar"] = df_v1["grammar"].str.lower()
    df_v2["grammar"] = df_v2["grammar"].str.lower()


    df = df_v2.merge(
        df_v1[["grammar", "seed", "v1_score"]],
        on=["grammar", "seed"],
        how="inner",
    )


    lambdas = sorted(df["lambda"].unique())

    plt.figure(figsize=(8, 6))

    print("\n===== Correlation between Living Metric v1 and v2 =====\n")

    for lambda_ in lambdas:
        sub = df[df["lambda"] == lambda_]

        x = sub["v1_score"].values
        y = sub["living_metric_score_value"].values

        r, p = pearsonr(x, y)

        print(
            f"lambda = {lambda_:>3.1f} | "
            f"Pearson r = {r: .3f} | "
            f"p-value = {p: .3e}"
        )

        plt.scatter(
            x,
            y,
            alpha=0.7,
            label=f"λ = {lambda_} (r = {r:.2f})",
        )

    plt.xlabel("Living Metric v1 score")
    plt.ylabel("Living Metric v2 score (normalised)")
    plt.title("Living Metric v1 vs v2 (by λ)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    lambda_focus = 0.4
    df_lambda = df[df["lambda"] == lambda_focus]

    plt.figure(figsize=(8, 6))

    for grammar in sorted(df_lambda["grammar"].unique()):
        sub = df_lambda[df_lambda["grammar"] == grammar]

        plt.scatter(
            sub["v1_score"],
            sub["living_metric_score_value"],
            alpha=0.8,
            label=grammar.capitalize(),
        )

    plt.xlabel("Living Metric v1 score")
    plt.ylabel("Living Metric v2 score (normalised)")
    plt.title(f"Living Metric v1 vs v2 (λ = {lambda_focus}) by Grammar")
    plt.legend(title="Grammar")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    calculate_and_plot_correlation()