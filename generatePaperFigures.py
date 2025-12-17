import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class FigureGenerator():
    def __init__(self):
        self.grammar_colours = {
            "Organic"   :   "#5fb474",
            "Grid"      :   "#cda1cb",
            "Hex"       :   "#ffd937",
            "Line"      :   "#5a9fcb",
        }
        # Keep the desired order of grammars
        self.grammar_order = list(self.grammar_colours.keys())

        self.metrics = {
            "mean_circuity" :   {
                "data_path"     : "mean_circuity.csv",
                "name"          : "Mean circuity"
            },

            "road_density"  :   {
                "data_path"     : "road_density.csv",
                "name"          : "Road density"
            },

            "living_metric" :   {
                "data_path"     : "living_metric_beta_5.csv",
                "name"          : "Living metric"

            }
        }
    
    def load_data(self, path):
        df = pd.read_csv(path, header=None, names = ["grammar","population","seed", "value"])
        return df

    def plot_scatter(self, save_path=None, show=True):
        circuity_df = self.load_data("mean_circuity.csv")
        circuity_df = circuity_df.rename(columns = {"value":"mean_circuity"})

        road_density_df = self.load_data("road_density.csv")
        road_density_df = road_density_df.rename(columns = {"value":"road_density"})

        merged_df = circuity_df.merge(
            road_density_df,
            on=["grammar", "seed", "population"],
            how="inner"
        )

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
        
        plt.xlabel("Road Density")
        plt.ylabel("Mean Circuity")
        plt.title("Road Density vs Mean Circuity by Grammar")
        plt.legend(title="Grammar", loc="upper left")
        plt.gca().invert_yaxis()

        plt.tight_layout()

        if show:
            plt.show()

        if save_path:
            plt.savefig(save_path)

    def plot_violins(self, metric, show=True, save_path = None):
        df = self.load_data(self.metrics[metric]["data_path"])
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
        plt.ylabel(self.metrics[metric]["name"])
        plt.subplots_adjust(left=-0.25)


        plt.tight_layout()

        if show:
            plt.show()

        if save_path:
            plt.savefig(save_path)

    def plot_metric_heatmap(self, show=True, save_path=None):
        # Compute median of each metric per grammar
        summary = {}
        for metric, details in self.metrics.items():
            df = self.load_data(details["data_path"])
            median_vals = df.groupby("grammar")["value"].median()
            summary[details["name"]] = median_vals

        # Desired row order
        row_order = ["Road density", "Living metric", "Mean circuity"]

        # Reorder rows in DataFrame
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

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8,3))
        sns.heatmap(
            normed,
            annot=heatmap_df.round(4),
            fmt="",
            cmap="RdYlGn",
            cbar=False,
            linewidths=2,
            linecolor="white",
            ax=ax
        )

        ax.hlines([1,2], *ax.get_xlim(), colors="black", linewidth=2)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        ax.set_ylabel("")
        ax.set_xlabel("")

        box = ax.get_position()
        ax.set_position([0.25, box.y0, 0.5, box.height])  
    
        if show:
            plt.show()
        if save_path:
            fig.savefig(save_path)


fig_gen = FigureGenerator()

fig_gen.plot_scatter(show=False, save_path = "road_density_vs_mean_circuity_scatter.pdf")

fig_gen.plot_metric_heatmap(show=False, save_path="results_table_heatmap.pdf")

fig_gen.plot_violins("road_density", show=False, save_path = "road_density_violins.pdf")
fig_gen.plot_violins("mean_circuity", show=False, save_path = "mean_circuity_violins.pdf")
fig_gen.plot_violins("living_metric", show=False, save_path = "living_metric_violins.pdf")
