import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

class FigureGenerator():
    def __init__(self):
        self.grammar_colours = {
            "organic"   :   "#5fb474",
            "grid"      :   "#cda1cb",
            "hex"       :   "#ffd937",
            "line"      :   "#5a9fcb",
        }

        # Keep the desired order of grammars
        self.grammar_order = list(self.grammar_colours.keys())

        self.metrics = {
            "mean_circuity" :   {
                "data_path"     : "new_metrics/mean_circuity.csv",
                "name"          : "Mean circuity"
            },

            "road_density"  :   {
                "data_path"     : "new_metrics/road_density.csv",
                "name"          : "Road density"
            },

            # "living_metric" :   {
            #     "data_path"     : "living_metric_beta_5.csv",
            #     "name"          : "Living metric"

            # },

            "living_metric_v2" :   {
                "data_path"     : "new_metrics/living_metric_v2_lambda_5.csv",
                "name"          : "Living metric"

            }

        }

        self.bad_line_seeds = [2,3,8,11,19,35,38,39,40,45,47,49]
    
        self.heatmap_colourmap = self.heatmap_colourmap()

    def load_data(self, path, exclude_seeds=None):
        df = pd.read_csv(
            path,
            header=None,
            names=["grammar", "population", "seed", "value"]
        )

        if exclude_seeds is not None:
            df = df[~df["seed"].isin(exclude_seeds)]

        return df

    def plot_beta_sensitivity(self, save_path=None, show=True, csv_file=""):
        plt.rcParams['text.usetex'] = True

        df = pd.read_csv(csv_file)
        data = []
        for beta in df["beta"].unique():
            for seed in df["seed"].unique():
                row_sum = df[(df["beta"]==beta)&(df["seed"]==seed)]["living_metric"].sum().item()
                for grammar in df["grammar"].unique():
                    val = df[(df["beta"]==beta)&(df["seed"]==seed)&(df["grammar"]==grammar)]["living_metric"].sum().item()
                    data.append({"beta": beta, "grammar": grammar, "seed": seed, "living_metric": val/row_sum})

        df2 = pd.DataFrame(data)
        print(df2)
        plt.figure(figsize=(6,4))
        g = sns.lineplot(
            data=df2,
            x="beta",
            y="living_metric",
            hue="grammar",
            style="grammar",
            palette=self.grammar_colours,   
        )
        g.set_xticks(range(20))
        plt.xlabel("Beta")
        plt.ylabel("Normalised Living Metric")
        plt.title("Normalised Living metric vs Beta")
        plt.legend(title="Grammar", loc="upper right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()


    def plot_scatter(self, save_path=None, show=True, exclude_seeds=None):
        plt.rcParams['text.usetex'] = True

        circuity_df = self.load_data(self.metrics["mean_circuity"]["data_path"], exclude_seeds=exclude_seeds)
        circuity_df = circuity_df.rename(columns = {"value":"mean_circuity"})

        road_density_df = self.load_data(self.metrics["road_density"]["data_path"], exclude_seeds=exclude_seeds)
        road_density_df = road_density_df.rename(columns = {"value":"road_density"})

        merged_df = circuity_df.merge(
            road_density_df,
            on=["grammar", "seed", "population"],
            how="inner"
        )

        print(merged_df)

        plt.figure(figsize=(6,4))
        sns.scatterplot(
            data=merged_df,
            x="road_density",
            y="mean_circuity",
            hue="grammar",  
            style="grammar", 
            palette=self.grammar_colours,   
            hue_order=self.grammar_order,   
            s=100,
            edgecolors="black"
        )

        plt.ylim(1.15,1.50)
        
        plt.xlabel("Road Density")
        plt.ylabel("Mean Circuity")
        plt.title("Road Density vs Mean Circuity by Grammar")
        plt.legend(title="Grammar", loc="upper right")
        plt.gca().invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()



    def plot_violins(self, metric, exclude_seeds = None, show=True, save_path = None):
        plt.rcParams['text.usetex'] = True

        df = self.load_data(self.metrics[metric]["data_path"], exclude_seeds=exclude_seeds)
        df = df.rename(columns={"value": metric})

        plt.figure(figsize=(3,6))

        # Violin plot
        sns.violinplot(
            x='grammar', 
            y=metric, 
            data=df, 
            inner=None, 
            palette=self.grammar_colours,
            order=self.grammar_order,
            edgecolor='black',
            hue = 'grammar'
        )


        sns.boxplot(
            x='grammar', 
            y=metric, 
            data=df, 
            width=0.15,
            showcaps=True, 
            boxprops={'facecolor':'None', 'edgecolor':'black'}, 
            whiskerprops={'color':'black'},
            capprops={'color':'black'},
            medianprops={'color':'black'},
            showfliers=False,
            order=self.grammar_order
        )

        if metric == "mean_circuity":
            plt.gca().invert_yaxis()

        plt.xlabel("Grammar")

        if metric == "living_metric_v2":
            plt.ylabel("Living metric (lambda=5)")
        else:
            plt.ylabel(self.metrics[metric]["name"])
        
        plt.subplots_adjust(left=-0.25)


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

    def plot_metric_heatmap(self, cmap, exclude_seeds = None, show=True, save_path=None):
        plt.rcParams['text.usetex'] = True
        
        # Load and summarize median values for each metric
        summary = {}
        for metric, details in self.metrics.items():
            df = self.load_data(details["data_path"], exclude_seeds=exclude_seeds)
            median_vals = df.groupby("grammar")["value"].median()
            summary[details["name"]] = median_vals

        # Desired row order
        row_order = ["Road density", "Living metric", "Mean circuity"]

        # Create DataFrame in desired order
        heatmap_df = pd.DataFrame(summary).T.loc[row_order, self.grammar_order]

        # Normalise each row for color mapping
        normed = heatmap_df.copy()
        for row in normed.index:
            vals = normed.loc[row]
            min_v, max_v = vals.min(), vals.max()
            if max_v > min_v:
                if row in ["Road density", "Living metric"]:
                    # higher is better → best = max = green
                    normed.loc[row] = (vals - min_v) / (max_v - min_v)
                else:
                    # Mean circuity → lower is better → best = min = green
                    normed.loc[row] = (vals - max_v) / (min_v - max_v)

        # --- Plot using stacked subplots ---
        fig, axs = plt.subplots(
            nrows=len(row_order),
            ncols=1,
            figsize=(5, 2.4),
            gridspec_kw={"hspace": 0.2}  # space between rows
        )

        for i, row in enumerate(row_order):
            ax = axs[i]
            sns.heatmap(
                normed.loc[[row]],
                annot=heatmap_df.loc[[row]].round(4),
                fmt=".4f",
                cmap=cmap,
                cbar=False,
                linewidths=2,
                linecolor="white",
                ax=ax,
                #annot_kws={"fontsize": 15},
            )

            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", va = "center") 


            if i == 0:
                # Only top row shows x-tick labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
                ax.xaxis.set_ticks_position("top")
                ax.xaxis.set_label_position("top")
            else:
                # Remove ticks completely for bottom rows
                ax.set_xticks([])

            # Keep y-axis ticks
            ax.set_ylabel("")
            ax.set_xlabel("")

        # Optional: adjust left/right positions
        fig.subplots_adjust(left=0.2, right=0.8)

        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()




    def plot_grammar_colourbar(self,cmap, show = False, save_path = None):
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(8.5, 1))
        fig.subplots_adjust(bottom=0.5)

        norm = plt.Normalize(0, 10)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.ax.tick_params(labelsize=20)
        cb1.set_ticks([0, 10])
        cb1.set_ticklabels(['Worst grammar for given metric', 'Best grammar for given metric'])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()




    def heatmap_colourmap(name="red_gold_green"):

        colours = [
            "#b50c00",   # red
            "#FFDD00",  # gold
            "#147b00",  # green
        ]
        return mcolors.LinearSegmentedColormap.from_list(name, colours)

def continuous_cmap_from_colours(self, colours, name="custom_continuous"):

    return mcolors.LinearSegmentedColormap.from_list(name, colours)


if __name__=="__main__":
    fig_gen = FigureGenerator()

    cmap = fig_gen.heatmap_colourmap
    fig_gen.plot_grammar_colourbar(cmap=cmap, show=False, save_path="resultsTableColourbar.pdf")


    # Generate figures based on CSV files
    fig_gen.plot_scatter(show=False, exclude_seeds=fig_gen.bad_line_seeds, save_path = "road_density_vs_mean_circuity_scatter.pdf")
    fig_gen.plot_metric_heatmap(show=False, cmap = cmap, exclude_seeds=fig_gen.bad_line_seeds, save_path="results_table_heatmap.pdf")
    fig_gen.plot_violins("road_density", show=False, exclude_seeds=fig_gen.bad_line_seeds, save_path = "road_density_violins.pdf")
    fig_gen.plot_violins("mean_circuity", show=False, exclude_seeds=fig_gen.bad_line_seeds, save_path = "mean_circuity_violins.pdf")
    fig_gen.plot_violins("living_metric_v2", show=False, exclude_seeds=fig_gen.bad_line_seeds, save_path = "living_metric_violins.pdf")
