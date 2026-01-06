import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

class FigureGeneratorV2:
    def __init__(self):
        # Grammar colours and order
        self.grammar_colours = {
            "organic": "#5fb474",
            "grid": "#cda1cb",
            "hex": "#ffd937",
            "line": "#5a9fcb",
        }
        self.grammar_order = list(self.grammar_colours.keys())




    def load_metric_file(self, path, long_format=False):
        """
        Load a CSV file. If long_format=False, assumes old format: grammar,seed,population,value.
        If long_format=True, converts wide CSV to long format for plotting.
        """
        df = pd.read_csv(path)
        if long_format:
            # Melt wide format
            value_cols = [c for c in df.columns if c not in ["grammar", "seed"]]
            df_long = df.melt(
                id_vars=["grammar", "seed"],
                value_vars=value_cols,
                var_name="metric",
                value_name="value"
            )
            # Extract numeric lambda values from column names
            df_long["lambda"] = df_long["metric"].str.extract(r"lambda_(.*)").astype(float)
            # Keep only living_metric_v2 columns
            df_long = df_long.drop(columns=["metric"])
            return df_long
        else:
            return df

    def plot_lambda_sensitivity(self, csv_file, show=True, save_path=None):
        """
        Plot sensitivity of living_metric_v2 vs lambda.
        """
        plt.rcParams['text.usetex'] = True
        df = self.load_metric_file(csv_file, long_format=True)

        # Normalize per (seed)
        data = []
        for lam in df["lambda"].unique():
            for seed in df["seed"].unique():
                row_sum = df[(df["lambda"]==lam)&(df["seed"]==seed)]["value"].sum()
                for grammar in df["grammar"].unique():
                    val = df[(df["lambda"]==lam)&(df["seed"]==seed)&(df["grammar"]==grammar)]["value"].sum()
                    data.append({"lambda": lam, "grammar": grammar, "seed": seed, "value": val/row_sum})

        df2 = pd.DataFrame(data)

        plt.figure(figsize=(6,4))
        sns.lineplot(
            data=df2,
            x="lambda",
            y="value",
            hue="grammar",
            style="grammar",
            palette=self.grammar_colours,
            ci=None
        )
        plt.xlabel("Lambda")
        plt.ylabel("Proportion of total living metric")
        plt.title("Lambda Sensitivity of Living Metric")
        plt.legend(title="Grammar", loc="upper right")
        plt.tight_layout()
        plt.ylim([0,0.6])

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()


if __name__ == "__main__":
    # Initialize figure generator
    fig_gen = FigureGeneratorV2()

    # File path
    csv_file = "lambda_sensitivity.csv"

    # 1️⃣ Plot lambda sensitivity for living_metric_v2
    fig_gen.plot_lambda_sensitivity(csv_file=csv_file, show=False, save_path="lambda_sensitivity.pdf")


