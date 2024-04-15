import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.ticker as plticker
import cityGenerator
import generateMaps as maps
import matplotlib as mpl

# Define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# Define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch

class ComplexityPlotter:
    def __init__(self):
        self.grammarList = ["Organic", "Grid", "Hex", "Line"]
        self.seedList = [0,1,2,3,4]
        self.populationList = [5000000]
        self.colourPalettes = {
            "Grid": sns.color_palette("PuRd", len(self.seedList)),
            "Hex": sns.color_palette("Wistia", len(self.seedList)),
            "Line": sns.color_palette("Blues_d", len(self.seedList)),
            "Organic": sns.color_palette("Greens_d", len(self.seedList))
        }

    def quadFuncThroughOrigin(self, x, a, b):
        return a * x**2 + b * x

    def createDataFrame(self, grammar, population, seed):
        path = f"statsAndFigs/complexityData/{grammar}/{grammar}Population{population}seed{seed}Complexity.txt"
        df = pd.read_csv(path, sep=',', header=None)
        df.columns = ['Network size (nodes)', 'Time taken to reach size (seconds)']
        return df

    def createListOfExperimentalData(self):
        dfList = []
        dfKeys = []
        for population in self.populationList:
            for grammar in self.grammarList:
                for seed in self.seedList:
                    path = f"statsAndFigs/complexityData/{grammar}/{grammar}Population{population}seed{seed}Complexity.txt"
                    df = pd.read_csv(path, sep=',', header=None)
                    df.columns = ['Network size (nodes)', 'Time taken to reach size (seconds)']
                    dfList.append(df)
                    dfKeys.append((f"{grammar}", f"{seed}"))
        return dfList, dfKeys

    def addQuadraticFitThroughOrigin(self, df):
        popt, pcov = curve_fit(self.quadFuncThroughOrigin, df['Network size (nodes)'], df['Time taken to reach size (seconds)'])
        df['Quadratic fit'] = self.quadFuncThroughOrigin(df['Network size (nodes)'], *popt)
        return df

    def createListOfExperimentalAndFittedData(self):

        dfList, dfKeys = self.createListOfExperimentalData()
        for i, df in enumerate(dfList):
            dfList[i] = self.addQuadraticFitThroughOrigin(df)
        return dfList, dfKeys

    def plotAllComplexityCombinations(self,show=True):
        dfList, dfKeys = self.createListOfExperimentalAndFittedData()

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')

        fig, ax = plt.subplots()
        ax.set_xlabel('Network size (nodes)')
        ax.set_ylabel('Time taken to reach size (seconds)')

        handles = []
        labels = []

        for grammar in self.grammarList:
            # Create a MulticolorPatch for each grammar
            colors = self.colourPalettes[grammar]
            handles.append(MulticolorPatch(colors))
            labels.append(r'\texttt{' + grammar + '}')

        for i, df in enumerate(dfList):
            grammar, seed = dfKeys[i]
            color = self.colourPalettes[grammar][int(seed)]
            ax.plot(df['Network size (nodes)'], df['Quadratic fit'], label=dfKeys[i], color=color)

        # Use a custom handler to display multicolor legend patches
        ax.legend(handles, labels, loc='upper left', 
                 handler_map={MulticolorPatch: MulticolorPatchHandler()}, 
                 bbox_to_anchor=(.125,.875),title='Grammar')
        
        plt.tight_layout()

        if show:
            plt.show()


    def plotSingleComplexityCombination(self, grammar, population, seed, show=True):
        df = self.createDataFrame(grammar, population, seed)
        
        df = self.addQuadraticFitThroughOrigin(df)

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')

        fig, ax = plt.subplots(figsize=(4, 8/3))
        ax.set_xlabel('Network size (nodes)')
        ax.set_ylabel('Time taken to \n reach size (seconds)')

        #set the ticks on the y axis to be every 500 seconds
        ax.set_yticks(np.arange(0, 3000, 500))

        ax.scatter(df['Network size (nodes)'], df['Time taken to reach size (seconds)'], label="Experimental data", marker="x", color = 'black', s=30)
        ax.plot(df['Network size (nodes)'], df['Quadratic fit'], label=None, color='black', linewidth=3)
        ax.plot(df['Network size (nodes)'], df['Quadratic fit'], label="Quadratic fit", color=self.colourPalettes[grammar][1], linewidth=2)

#self.colourPalettes[grammar][0])

        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

    def saveComplexitySubplots(self, population, seed):
        for grammar in self.grammarList:
            self.plotSingleCombination(grammar, population, seed, show=False)
            plt.savefig(f"statsAndFigs/figs/complexityFigs/{grammar}Population{population}seed{seed}Complexity.pdf")

    def saveAllComplexityCombinations(self):
        self.plotAllCombinations(show=False)
        plt.savefig(f"statsAndFigs/figs/complexityFigs/AllCombinationsComplexity.pdf")


class metricPlotter:
    def __init__(self):
        self.grammarList = ["Organic", "Grid", "Hex", "Line"]
        self.seedList = [0,1,2,3,4]
        self.populationList = [000000]
        self.colourPalettes = {
            "Grid": sns.color_palette("PuRd", 5),
            "Hex": sns.color_palette("Wistia", 5),
            "Line": sns.color_palette("Blues_d", 5),
            "Organic": sns.color_palette("Greens_d", 5)
        }

    def plotAllCircuityBoxplots(self, circuityPath, show=True):

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(circuityPath, sep=',', header=None)    
        df.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']

        fig, ax = plt.subplots(figsize=(7, 10))

        palette = {grammar: self.colourPalettes[grammar][1] for grammar in df['Grammar'].unique()}

        order = ['Organic', 'Grid', 'Hex', 'Line']

        sns.boxplot(x='Grammar', y='Average Circuity', data=df, ax=ax, palette=palette, linewidth=0.7, fliersize=2, boxprops={'edgecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, flierprops={'marker': 'x', 'markersize': 2, 'markeredgecolor': 'black'}, capprops={'color': 'black'}, order=order)

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Mean circuity')  

        # Set the x-axis labels to the 'monospace' font
        ax.set_xticklabels(r'\texttt{' + grammar + '}' for grammar in order)

        plt.tight_layout()

        if show:
            plt.show()

    def plotSingleCircuityBoxplot(self, circuityPath, grammar, show=True):

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(circuityPath, sep=',', header=None)    
        df.columns = ['Grammar', 'Population', 'Seed', 'Average circuity']
        df = df[df['Grammar'] == grammar]

        fig, ax = plt.subplots(figsize=(1.8, 3))
        plt.subplots_adjust(left=0.5, right=0.95, bottom=0.05, top=0.95)
        #ax.set_xlabel('Grammar')
        #ax.set_ylabel('Average circuity')


        palette = {grammar: self.colourPalettes[grammar][1] for grammar in df['Grammar'].unique()}

        sns.boxplot(x='Grammar', y='Average circuity', data=df, ax=ax, palette=palette, boxprops={'edgecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, flierprops={'marker': 'x', 'markersize': 3, 'markeredgecolor': 'black'}, capprops={'color': 'black'})
        ax.set_xticklabels([])

        #plt.tight_layout()


        ax.yaxis.set_label_coords(-0.8, 0.5)

        

        if show:
            plt.show()

    def saveCircuitySubplots(self, circuityPath):
        for grammar in self.grammarList:
            self.plotSingleCircuityBoxplot(circuityPath, grammar, show=False)
            plt.savefig(f"statsAndFigs/figs/circuityPlots/{grammar}AverageCircuity.pdf")

    def saveAllCircuityBoxplotsOnOne(self, circuityPath):
        self.plotAllCircuityBoxplots(circuityPath, show=False)
        plt.savefig(f"statsAndFigs/figs/circuityPlots/allCircuityBoxplots.pdf")


    def plotCircuityScatter(self,circuityPath, show=True,figSize=(7, 4)):
        #this will be the same as the plotAllBoxplots function but with a scatter plot instead of a boxplot.

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(circuityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']

        # Convert 'Grammar' to a categorical type with the specified order
        df['Grammar'] = pd.Categorical(df['Grammar'], categories=['Organic', 'Grid', 'Hex', 'Line'], ordered=True)

        # Sort dataframe by 'Grammar'
        df.sort_values('Grammar', inplace=True)

        fig, ax = plt.subplots(figsize=figSize)

        for grammar in df['Grammar'].unique():
            grammar_df = df[df['Grammar'] == grammar]
            ax.scatter(grammar_df['Grammar'], grammar_df['Average Circuity'], c=self.colourPalettes[grammar][0], marker='o', s=20, edgecolors='black')

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Mean circuity')

                        # Set x-tick labels in LaTeX texttt mode
        ax.set_xticklabels([r'\texttt{' + grammar + '}' for grammar in df['Grammar'].unique()])

        plt.tight_layout()

        if show:
            plt.show()

    def saveCircuityScatter(self, circuityPath):
        self.plotCircuityScatter(circuityPath, show=False)
        plt.savefig(f"statsAndFigs/figs/circuityPlots/allCircuityScatter.pdf")

    def plotSingleDensityBoxplot(self, densityPath, grammar, show=True):
    
        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(densityPath, sep=',', header=None)    
        df.columns = ['Grammar', 'Population', 'Seed', 'Population density']
        df = df[df['Grammar'] == grammar]

        fig, ax = plt.subplots(figsize=(1.8, 3))
        plt.subplots_adjust(left=0.5, right=0.95, bottom=0.05, top=0.95)
        #ax.set_xlabel('Grammar')
        #ax.set_ylabel('Average circuity')


        palette = {grammar: self.colourPalettes[grammar][1] for grammar in df['Grammar'].unique()}

        sns.boxplot(x='Grammar', y='Population density', data=df, ax=ax, palette=palette, boxprops={'edgecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, flierprops={'marker': 'x', 'markersize': 3, 'markeredgecolor': 'black'}, capprops={'color': 'black'})
        ax.set_xticklabels([])

        #plt.tight_layout()


        ax.yaxis.set_label_coords(-0.8, 0.5)

        if show:
            plt.show()

    def plotAllDensityBoxplots(self, densityPath, show=True):
            
        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(densityPath, sep=',', header=None)    
        df.columns = ['Grammar', 'Population', 'Seed', 'Population density']

        fig, ax = plt.subplots(figsize=(7, 10))

        palette = {grammar: self.colourPalettes[grammar][1] for grammar in df['Grammar'].unique()}

        order = ['Organic', 'Grid', 'Hex', 'Line']

        sns.boxplot(x='Grammar', y='Population density', data=df, ax=ax, palette=palette, linewidth=0.7, fliersize=2, boxprops={'edgecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, flierprops={'marker': 'x', 'markersize': 2, 'markeredgecolor': 'black'}, capprops={'color': 'black'}, order=order)

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Population density')  

        # Set the x-axis labels to the 'monospace' font
        ax.set_xticklabels(r'\texttt{' + grammar + '}' for grammar in order)

        plt.tight_layout()

        if show:
            plt.show()

    def saveAllDensityBoxplotsOnOne(self, densityPath):
        self.plotAllDensityBoxplots(densityPath, show=False)
        plt.savefig(f"statsAndFigs/figs/densityPlots/allPopulationDensityBoxplots.pdf")

    def saveDensitySubplots(self, densityPath):
        for grammar in self.grammarList:
            self.plotSingleDensityBoxplot(densityPath, grammar, show=False)
            plt.savefig(f"statsAndFigs/figs/densityPlots/{grammar}PopulationDensity.pdf")

    def plotDensityScatter(self,densityPath, show=True):
        #this will be the same as the plotAllBoxplots function but with a scatter plot instead of a boxplot.

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(densityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Population density']

        # Convert 'Grammar' to a categorical type with the specified order
        df['Grammar'] = pd.Categorical(df['Grammar'], categories=['Organic', 'Grid', 'Hex', 'Line'], ordered=True)

        # Sort dataframe by 'Grammar'
        df.sort_values('Grammar', inplace=True)

        fig, ax = plt.subplots(figsize=(7, 4))

        for grammar in df['Grammar'].unique():
            grammar_df = df[df['Grammar'] == grammar]
            ax.scatter(grammar_df['Grammar'], grammar_df['Population density'], c=self.colourPalettes[grammar][1], marker='o', s=20, edgecolors='black')

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Population density')

        plt.tight_layout()

        if show:
            plt.show()

    def saveDensityScatter(self, densityPath):
        self.plotDensityScatter(densityPath, show=False)
        plt.savefig(f"statsAndFigs/figs/densityPlots/allPopulationDensityScatter.pdf")

    def calculateMeanPopulationDensity(self, densityPath, grammar):
        # calculate the mean population density for the given grammar
        df = pd.read_csv(densityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Population density']
        df = df[df['Grammar'] == grammar]
        return df['Population density'].mean()
    
    def calculateMedianPopulationDensity(self, densityPath, grammar):
        df = pd.read_csv(densityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Population density']
        df = df[df['Grammar'] == grammar]
        
        Q1 = df['Population density'].quantile(0.25)
        Q3 = df['Population density'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['Population density'] >= lower_bound) & (df['Population density'] <= upper_bound)]

        return filtered_df['Population density'].median()
    
    def calculatePopulationDensityIQR(self, densityPath, grammar):
        # calculate the interquartile range of the population density for the given grammar
        df = pd.read_csv(densityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Population density']
        df = df[df['Grammar'] == grammar]
        return df['Population density'].quantile(0.75) - df['Population density'].quantile(0.25)
    
    def calculateMedianCircuity(self, circuityPath, grammar):
        """
        df = pd.read_csv(circuityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']
        df = df[df['Grammar'] == grammar]
        
        Q1 = df['Average Circuity'].quantile(0.25)
        Q3 = df['Average Circuity'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['Average Circuity'] >= lower_bound) & (df['Average Circuity'] <= upper_bound)]

        return filtered_df['Average Circuity'].median()
        """

        df = pd.read_csv(circuityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']
        df = df[df['Grammar'] == grammar]

        return df['Average Circuity'].median()

    def calculateCircuityIQR(self, circuityPath, grammar):
        # calculate the interquartile range of the circuity for the given grammar
        df = pd.read_csv(circuityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']
        df = df[df['Grammar'] == grammar]
        return df['Average Circuity'].quantile(0.75) - df['Average Circuity'].quantile(0.25)
    
    def findMinMaxCircuity(self, circuityPath, grammar, minOrMax):
        # find the minimum or maximum circuity for the given grammar, and the seed that produced it
        df = pd.read_csv(circuityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']
        df = df[df['Grammar'] == grammar]
        if minOrMax == "min":
            return df['Average Circuity'].min(), df.loc[df['Average Circuity'].idxmin()]['Seed']
        elif minOrMax == "max":
            return df['Average Circuity'].max(), df.loc[df['Average Circuity'].idxmax()]['Seed']
        else:
            return None

    def plotAllRoadDensityBoxplots(self, roadDensityPath, show=True):
            
        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(roadDensityPath, sep=',', header=None)    
        df.columns = ['Grammar', 'Population', 'Seed', 'Road density']

        fig, ax = plt.subplots(figsize=(7, 10))

        palette = {grammar: self.colourPalettes[grammar][1] for grammar in df['Grammar'].unique()}

        order = ['Organic', 'Grid', 'Hex', 'Line']

        sns.boxplot(x='Grammar', y='Road density', data=df, ax=ax, palette=palette, linewidth=0.7, fliersize=2, boxprops={'edgecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, flierprops={'marker': 'x', 'markersize': 2, 'markeredgecolor': 'black'}, capprops={'color': 'black'}, order=order)

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Road density (units of road length per unit area)') 

        # Set the x-axis labels to the 'monospace' font
        ax.set_xticklabels(r'\texttt{' + grammar + '}' for grammar in order) 

        plt.tight_layout()

        if show:
            plt.show()

    def plotRoadDensityScatter(self,roadDensityPath, show=True, figSize=(7, 4)):

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(roadDensityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Road density']

        # Convert 'Grammar' to a categorical type with the specified order
        df['Grammar'] = pd.Categorical(df['Grammar'], categories=['Organic', 'Grid', 'Hex', 'Line'], ordered=True)

        # Sort dataframe by 'Grammar'
        df.sort_values('Grammar', inplace=True)

        fig, ax = plt.subplots(figsize=figSize)

        for grammar in df['Grammar'].unique():
            grammar_df = df[df['Grammar'] == grammar]
            ax.scatter(grammar_df['Grammar'], grammar_df['Road density'], c=self.colourPalettes[grammar][1], marker='o', s=20, edgecolors='black')

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Road density')

                # Set x-tick labels in LaTeX texttt mode
        ax.set_xticklabels([r'\texttt{' + grammar + '}' for grammar in df['Grammar'].unique()])

        plt.tight_layout()

        if show:
            plt.show()


    def saveAllRoadDensityBoxplotsOnOne(self, roadDensityPath):
        self.plotAllRoadDensityBoxplots(roadDensityPath, show=False)
        plt.savefig(f"statsAndFigs/figs/roadDensityPlots/allRoadDensityBoxplots.pdf")

    def calculateRoadDensityIQR(self, roadDensityPath, grammar):
        # calculate the interquartile range of the road density for the given grammar
        df = pd.read_csv(roadDensityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Road density']
        df = df[df['Grammar'] == grammar]
        return df['Road density'].quantile(0.75) - df['Road density'].quantile(0.25)

    def calculateMedianRoadDensity(self, roadDensityPath, grammar):
        """
        df = pd.read_csv(roadDensityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Road density']
        df = df[df['Grammar'] == grammar]
        
        Q1 = df['Road density'].quantile(0.25)
        Q3 = df['Road density'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['Road density'] >= lower_bound) & (df['Road density'] <= upper_bound)]

        return filtered_df['Road density'].median()
        """

        df = pd.read_csv(roadDensityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Road density']
        df = df[df['Grammar'] == grammar]
        
        return df['Road density'].median()

    def findMinMaxRoadDensity(self, roadDensityPath, grammar, minOrMax):
        # find the minimum or maximum road density for the given grammar, and the seed that produced it
        df = pd.read_csv(roadDensityPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Road density']
        df = df[df['Grammar'] == grammar]
        if minOrMax == "min":
            return df['Road density'].min(), df.loc[df['Road density'].idxmin()]['Seed']
        elif minOrMax == "max":
            return df['Road density'].max(), df.loc[df['Road density'].idxmax()]['Seed']
        else:
            return None

    def plotRoadDensityVsMeanCircuity(self, roadDensityPath, circuityPath, show=True):
        # plot a scatter of road density against mean circuity. the colour of the points will be determined by the grammar, with color=self.colourPalettes[grammar][1]
        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        dfRoadDensity = pd.read_csv(roadDensityPath, sep=',', header=None)
        dfRoadDensity.columns = ['Grammar', 'Population', 'Seed', 'Road density']
        dfCircuity = pd.read_csv(circuityPath, sep=',', header=None)
        dfCircuity.columns = ['Grammar', 'Population', 'Seed', 'Average Circuity']

        # Convert 'Grammar' to a categorical type with the specified order
        dfRoadDensity['Grammar'] = pd.Categorical(dfRoadDensity['Grammar'], categories=['Organic', 'Grid', 'Hex', 'Line'], ordered=True)
        dfCircuity['Grammar'] = pd.Categorical(dfCircuity['Grammar'], categories=['Organic', 'Grid', 'Hex', 'Line'], ordered=True)

        # Sort dataframe by 'Grammar'
        dfRoadDensity.sort_values('Grammar', inplace=True)
        dfCircuity.sort_values('Grammar', inplace=True)

        fig, ax = plt.subplots(figsize=(7, 4))

        for grammar in dfRoadDensity['Grammar'].unique():
            grammar_df = dfRoadDensity[dfRoadDensity['Grammar'] == grammar]
            ax.scatter(grammar_df['Road density'], dfCircuity[dfCircuity['Grammar'] == grammar]['Average Circuity'], c=self.colourPalettes[grammar][1], marker='o', s=20, edgecolors='black', label=r'\texttt{' + grammar + '}')

        ax.set_xlabel('Road density (units of road length per unit area)')
        ax.set_ylabel('Mean circuity')

        #ax.invert_yaxis()  # This line will reverse the y-axis

        ax.legend()  # This line will add the legend

        plt.tight_layout()

        if show:
            plt.show()

    def saveRoadDensityVsMeanCircuity(self, roadDensityPath, circuityPath):
        self.plotRoadDensityVsMeanCircuity(roadDensityPath, circuityPath, show=False)
        plt.savefig(f"statsAndFigs/figs/roadDensityVsMeanCircuity.pdf")

    def plotAllLivingMetricBoxplots(self, livingMetricPath, show=True):

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(livingMetricPath, sep=',', header=None)    
        df.columns = ['Grammar', 'Population', 'Seed', 'Living ratio']

        fig, ax = plt.subplots(figsize=(7, 10))

        palette = {grammar: self.colourPalettes[grammar][1] for grammar in df['Grammar'].unique()}

        # Specify the order of the boxplots
        order = ['Organic', 'Grid', 'Hex', 'Line']

        sns.boxplot(x='Grammar', y='Living ratio', data=df, ax=ax, palette=palette, linewidth=0.7, fliersize=2, boxprops={'edgecolor': 'black'}, medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, flierprops={'marker': 'x', 'markersize': 2, 'markeredgecolor': 'black'}, capprops={'color': 'black'}, order=order)

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Living metric')  

        # Set the x-axis labels to the 'monospace' font
        ax.set_xticklabels(r'\texttt{' + grammar + '}' for grammar in order)

        plt.tight_layout()

        if show:
            plt.show()

    def saveAllLivingMetricBoxplotsOnOne(self, livingMetricPath):
        self.plotAllLivingMetricBoxplots(livingMetricPath, show=False)
        plt.savefig(f"statsAndFigs/figs/livingMetricPlots/allLivingMetricBoxplots.pdf")

    def calculateMedianLivingMetric(self, livingMetricPath, grammar):
        """
        df = pd.read_csv(livingMetricPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Living ratio']
        df = df[df['Grammar'] == grammar]
        
        Q1 = df['Living ratio'].quantile(0.25)
        Q3 = df['Living ratio'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df['Living ratio'] >= lower_bound) & (df['Living ratio'] <= upper_bound)]

        return filtered_df['Living ratio'].median()
        """
        df = pd.read_csv(livingMetricPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Living ratio']
        df = df[df['Grammar'] == grammar]
        
        return df['Living ratio'].median()

    def plotHeatmapTableOfMedians(self, show=True):
        plt.rcParams['text.usetex'] = True

        medianDF = pd.DataFrame(columns=['Grammar', 'Road density', 'Living metric', 'Mean circuity'])

        for grammar in self.grammarList:
            new_row = pd.DataFrame({'Grammar': [grammar],  
                                    'Road density': [self.calculateMedianRoadDensity("statsAndFigs/roadDensity.txt", grammar)], 
                                    'Living metric': [self.calculateMedianLivingMetric("statsAndFigs/livingMetric5.txt", grammar)],
                                    'Mean circuity': [self.calculateMedianCircuity("statsAndFigs/averageCircuity.txt", grammar)]})
            medianDF = pd.concat([medianDF, new_row], ignore_index=True)

        # Keep a copy of the original medianDF for annotation
        annotDF = medianDF.copy()

        # Normalize each row for the colors
        medianDF['Mean circuity'] = (medianDF['Mean circuity'] - medianDF['Mean circuity'].min()) / (medianDF['Mean circuity'].max() - medianDF['Mean circuity'].min())
        medianDF['Road density'] = (medianDF['Road density'] - medianDF['Road density'].min()) / (medianDF['Road density'].max() - medianDF['Road density'].min())
        medianDF['Living metric'] = (medianDF['Living metric'] - medianDF['Living metric'].min()) / (medianDF['Living metric'].max() - medianDF['Living metric'].min())

        # Reverse the colors for the 'Circuity' column
        medianDF['Mean circuity'] = 1 - medianDF['Mean circuity']

        # Transpose the DataFrame
        medianDF = medianDF.set_index('Grammar').T
        annotDF = annotDF.set_index('Grammar').T

        # Create a heatmap table of the median values
        fig, ax = plt.subplots(figsize=(5, 12/5))
        sns.heatmap(medianDF, annot=annotDF, cmap='RdYlGn', center=0.5, linewidths=0.5, linecolor='black', cbar=False, fmt=".4f")
        ax.set_xticklabels([f'\\texttt{{{grammar}}}' for grammar in medianDF.columns], rotation=0)
        plt.yticks(rotation=45)

        ax.xaxis.tick_top() 

        # Set the xlabel and its position
        ax.set_xlabel("")
        ax.xaxis.set_label_position('top')

        plt.tight_layout()

        ax.set_xlim([0, len(medianDF.columns) + 0.9])

        if show:
            plt.show()

    def saveHeatmapTableOfMedians(self):
        self.plotHeatmapTableOfMedians(show=False)
        plt.savefig("statsAndFigs/figs/resultsTableHeatmap.pdf", bbox_inches='tight')

    def plotColourBar(self):
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(8.5, 1))
        fig.subplots_adjust(bottom=0.5)

        norm = plt.Normalize(0, 10)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap='RdYlGn_r', norm=norm, orientation='horizontal')
        cb1.ax.tick_params(labelsize=20)
        cb1.set_ticks([0, 10])
        cb1.set_ticklabels(['Best grammar for given metric', 'Worst grammar for given metric'])

        plt.tight_layout()

        #plt.show()
        plt.savefig("statsAndFigs/figs/resultsTableColourbar.pdf", bbox_inches='tight')

    def plotLivingMetricScatter(self, livingMetricPath, show=True, figSize=(7, 4)):

        plt.rcParams['text.usetex'] = True
        sns.set_theme(style='darkgrid')
        df = pd.read_csv(livingMetricPath, sep=',', header=None)
        df.columns = ['Grammar', 'Population', 'Seed', 'Living ratio']

        # Convert 'Grammar' to a categorical type with the specified order
        df['Grammar'] = pd.Categorical(df['Grammar'], categories=['Organic', 'Grid', 'Hex', 'Line'], ordered=True)

        # Sort dataframe by 'Grammar'
        df.sort_values('Grammar', inplace=True)

        fig, ax = plt.subplots(figsize=figSize)

        for grammar in df['Grammar'].unique():
            grammar_df = df[df['Grammar'] == grammar]
            ax.scatter(grammar_df['Grammar'], grammar_df['Living ratio'], c=self.colourPalettes[grammar][1], marker='o', s=20, edgecolors='black')

        ax.set_xlabel('Grammar')
        ax.set_ylabel('Living metric')

        # Set x-tick labels in LaTeX texttt mode
        ax.set_xticklabels([r'\texttt{' + grammar + '}' for grammar in df['Grammar'].unique()])

        plt.tight_layout()

        if show:
            plt.show()

    def plotSmallHeatmapOfResults(self, show=True):
        plt.rcParams['text.usetex'] = True

        medianDF = pd.DataFrame(columns=['Grammar', 'Road density', 'Living metric', 'Mean circuity'])

        for grammar in self.grammarList:
            new_row = pd.DataFrame({'Grammar': [grammar],  
                                    'Road density': [self.calculateMedianRoadDensity("statsAndFigs/roadDensity.txt", grammar)], 
                                    'Living metric': [self.calculateMedianLivingMetric("statsAndFigs/livingMetric5.txt", grammar)],
                                    'Mean circuity': [self.calculateMedianCircuity("statsAndFigs/averageCircuity.txt", grammar)]})
            medianDF = pd.concat([medianDF, new_row], ignore_index=True)

        # Keep a copy of the original medianDF for annotation
        annotDF = medianDF.copy()

        # Normalize each row for the colors
        medianDF['Mean circuity'] = (medianDF['Mean circuity'] - medianDF['Mean circuity'].min()) / (medianDF['Mean circuity'].max() - medianDF['Mean circuity'].min())
        medianDF['Road density'] = (medianDF['Road density'] - medianDF['Road density'].min()) / (medianDF['Road density'].max() - medianDF['Road density'].min())
        medianDF['Living metric'] = (medianDF['Living metric'] - medianDF['Living metric'].min()) / (medianDF['Living metric'].max() - medianDF['Living metric'].min())

        # Reverse the colors for the 'Circuity' column
        medianDF['Mean circuity'] = 1 - medianDF['Mean circuity']

        # Transpose the DataFrame
        medianDF = medianDF.set_index('Grammar').T
        annotDF = annotDF.set_index('Grammar').T

        # Create a heatmap table of the median values
        fig, ax = plt.subplots(figsize=(3, 1.5))
        sns.heatmap(medianDF, annot=annotDF, cmap='RdYlGn', center=0.5, linewidths=0.5, linecolor='black', cbar=False, fmt=".4f")
        ax.set_xticklabels([f'\\texttt{{{grammar}}}' for grammar in medianDF.columns], rotation=0)
        plt.yticks(rotation=0)

        ax.xaxis.tick_top() 

        # Set the xlabel and its position
        ax.set_xlabel("")
        ax.xaxis.set_label_position('top')

        plt.tight_layout()

        if show:
            plt.show()
        
plotter = metricPlotter()
cityGen = cityGenerator.CityGenerator()

#plotter.plotAllCombinations()
#plotter.plotSingleCombination("Hex", 5000000, 0)
#plotter.saveAllCombinations()
#plotter.saveSubplots(5000000, 0)

#plotter.plotAllBoxplots("statsAndFigs/averageCircuity.txt")
#plotter.saveAllCircuityBoxplots("statsAndFigs/averageCircuity.txt")

#plotter.plotCircuityScatter("statsAndFigs/averageCircuity.txt")
#plotter.saveCircuityScatter("statsAndFigs/averageCircuity.txt")

#plotter.plotSingleBoxplot("statsAndFigs/averageCircuity.txt", "Hex")
#plotter.saveCircuitySubplots("statsAndFigs/averageCircuity.txt")

#plotter.plotSingleDensityBoxplot("statsAndFigs/populationDensity.txt", "Organic")
#plotter.saveDensitySubplots("statsAndFigs/populationDensity.txt")

#plotter.plotDensityScatter("statsAndFigs/populationDensity.txt")
#plotter.saveDensityScatter("statsAndFigs/populationDensity.txt")

#print(plotter.calculateMeanPopulationDensity("statsAndFigs/populationDensity.txt", "Line"))
#print(plotter.calculatePopulationDensityIQR("statsAndFigs/populationDensity.txt", "Line"))
#print(plotter.calculateMedianPopulationDensity("statsAndFigs/populationDensity.txt", "Hex"))

#print(plotter.calculateMedianCircuity("statsAndFigs/averageCircuity.txt", "Organic"))

#print the minimum and maximum circuity for the Organic grammar
#print(plotter.findMinMaxCircuity("statsAndFigs/averageCircuity.txt", "Organic", "min"))
#print(plotter.findMinMaxCircuity("statsAndFigs/averageCircuity.txt", "Organic", "max"))

#plotter.plotAllCircuityBoxplots("statsAndFigs/averageCircuity.txt")
#plotter.plotAllDensityBoxplots("statsAndFigs/populationDensity.txt")

#plotter.saveAllCircuityBoxplotsOnOne("statsAndFigs/averageCircuity.txt")
#plotter.saveAllDensityBoxplotsOnOne("statsAndFigs/populationDensity.txt")

#plotter.plotAllRoadDensityBoxplots("statsAndFigs/roadDensity.txt")
#plotter.saveAllRoadDensityBoxplotsOnOne("statsAndFigs/roadDensity.txt")

#print(plotter.calculateMedianRoadDensity("statsAndFigs/roadDensity.txt", "Organic"))
#print(plotter.calculateRoadDensityIQR("statsAndFigs/roadDensity.txt", "Line"))

#print(plotter.findMinMaxRoadDensity("statsAndFigs/roadDensity.txt", "Organic", "max"))

#print(plotter.calculateCircuityIQR("statsAndFigs/averageCircuity.txt", "Line"))

#plotter.saveRoadDensityVsMeanCircuity("statsAndFigs/roadDensity.txt", "statsAndFigs/averageCircuity.txt")

#plotter.plotCircuityScatter("statsAndFigs/averageCircuity.txt", figSize=(3, 1.65), show=False)
#plt.savefig(f"statsAndFigs/figs/circuityPlots/smallCircuityScatter.jpg", dpi=800, bbox_inches='tight')

#plotter.plotRoadDensityScatter("statsAndFigs/roadDensity.txt", figSize=(3, 1.65), show=False)
#plt.savefig(f"statsAndFigs/figs/roadDensityPlots/smallRoadDensityScatter.jpg", dpi=800, bbox_inches='tight')

#plotter.plotLivingMetricScatter("statsAndFigs/livingMetric5.txt", figSize=(3, 1.65), show=False)
#plt.savefig(f"statsAndFigs/figs/livingMetricPlots/smallLivingMetricScatter.jpg", dpi=800, bbox_inches='tight')

#plotter.plotAllRoadDensityBoxplots("statsAndFigs/roadDensity.txt")
#plotter.plotAllCircuityBoxplots("statsAndFigs/averageCircuity.txt")
#plotter.plotAllLivingMetricBoxplots("statsAndFigs/livingMetric5.txt")

#plotter.plotHeatmapTableOfMedians()
#plotter.saveHeatmapTableOfMedians()
#plotter.plotColourBar()

#plotter.saveAllCircuityBoxplotsOnOne("statsAndFigs/averageCircuity.txt")
#plotter.saveAllRoadDensityBoxplotsOnOne("statsAndFigs/roadDensity.txt")
#plotter.saveAllLivingMetricBoxplotsOnOne("statsAndFigs/livingMetric5.txt")

plotter.plotSmallHeatmapOfResults(show=False)
plt.savefig("statsAndFigs/figs/resultsTableSmallHeatmap.jpeg", bbox_inches='tight', dpi=800)