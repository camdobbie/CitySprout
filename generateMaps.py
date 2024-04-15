import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, MultiPolygon
from metrics import alpha_shape, headTailBreaks, calculateRoadBetweennessCentrality
import pickle
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import colors
import math


plasmaCmap = plt.cm.plasma
plasmaCmapV2 = mcolors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=plasmaCmap.name, a=0.0, b=0.9),
    plasmaCmap(np.linspace(0.0, 0.9, 256))
) # this Cmap is the same as the original plasma,  however it is truncated at 0.9, so that the last colour is not too pale
# create a plasmaCmapV3 that removes the bottom 40% and the top 10% of the plasma map
plasmaCmapV3 = mcolors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=plasmaCmap.name, a=0.4, b=0.9),
    plasmaCmap(np.linspace(0.4, 0.9, 256))
)
customCmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black","blue","magenta","red","gold"])

colourPalettes = {
            "Grid": sns.color_palette("PuRd", 5),
            "Hex": sns.color_palette("Wistia", 5),
            "Line": sns.color_palette("Blues_d", 5),
            "Organic": sns.color_palette("Greens_d", 5)
        }

def plotCityBlackWithAlphaShape(G, alpha=20, showNodes = False, nodeLabelType = None, edgeLabelType = None, show=True, savePath=None, figSize=(3,3), roadWidthType = "Default", roadWidth = 1, boundaryWidth = 2):
    fig, ax = plt.subplots(figsize=figSize)#figsize=(10, 10))  
    ax.set_aspect('equal')

    if showNodes:
        node_size = 10
    else:
        node_size = 0
    if nodeLabelType == "Node Type":
        with_labels = True
        labels=nx.get_node_attributes(G, 'nodeType')
    elif nodeLabelType == "Node Number":
        with_labels = True
        labels = {node: node for node in G.nodes()}
    elif nodeLabelType == "Road Type":
        with_labels = True
        labels=nx.get_node_attributes(G, 'roadType')
    else:
        with_labels = False
        labels=None

    edges = G.edges()

    if roadWidthType == "Default":
        edge_widths = [1 if G[u][v]['weight'] == 1 else 0.8 if G[u][v]['weight'] == 0.5 else 0.5 for u, v in edges]
    elif roadWidthType == "Constant":
        edge_widths = [roadWidth for u, v in edges]

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edges(G, pos, edge_color='black', width=edge_widths, ax=ax)  # Draw edges with outline
    if edgeLabelType == "Edge Weight":
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    if edgeLabelType == "Road Number":
        edge_labels = nx.get_edge_attributes(G, 'roadNumber')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)

    # Calculate the alpha shape
    polygon, edge_points = alpha_shape(G, alpha)

    # Ensure the polygon is a MultiPolygon
    if isinstance(polygon, Polygon):
        polygon = MultiPolygon([polygon])

    # Iterate over each polygon in the MultiPolygon
    for poly in polygon.geoms:
        # Get the vertices of the alpha shape
        hull_points = list(poly.exterior.coords)

        # Create a Polygon patch
        hull_patch = patches.Polygon(hull_points, fill=None, edgecolor='red', linewidth=boundaryWidth)

        # Add the patch to the Axes
        ax.add_patch(hull_patch)

    if with_labels:
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
    plt.axis('off')  # Turn on the axes
    plt.grid(False)  # Add a grid
    if show:
        plt.show()
    if savePath:
        plt.savefig(savePath, bbox_inches='tight', dpi=800)

def plotCityBlack(G, showNodes = False, nodeLabelType = None, edgeLabelType = None, show = True, savePath = None, figsize=(10, 10), xlim = None, ylim = None):

    fig, ax = plt.subplots(figsize=figsize)  
    ax.set_aspect('equal')

    if showNodes:
        node_size = 10
    else:
        node_size = 0
    if nodeLabelType == "Node Type":
        with_labels = True
        labels=nx.get_node_attributes(G, 'nodeType')
    elif nodeLabelType == "Node Number":
        with_labels = True
        labels = {node: node for node in G.nodes()}
    elif nodeLabelType == "Road Type":
        with_labels = True
        labels=nx.get_node_attributes(G, 'roadType')
    else:
        with_labels = False
        labels=None

    edges = G.edges()
    edge_widths = [1 if G[u][v]['weight'] == 1 else 0.8 if G[u][v]['weight'] == 0.5 else 0.5 for u, v in edges]

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edges(G, pos, edge_color='black', width=edge_widths, ax=ax)  # Draw edges with outline
    if edgeLabelType == "Edge Weight":
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    if edgeLabelType == "Road Number":
        edge_labels = nx.get_edge_attributes(G, 'roadNumber')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    if showNodes:
        nx.draw_networkx_nodes(G, pos, node_size=node_size + 10, ax=ax, node_color="black")

    rgb_color = colors.rgb2hex(colourPalettes["Organic"][1])
    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax, node_color=rgb_color)

    if with_labels:
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
    plt.axis('off')  # Turn on the axes
    plt.grid(False)  # Add a grid

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if show:
        plt.show()
    if savePath:
        plt.savefig(savePath, bbox_inches='tight')


def plotRoadsByBetweennessCentrality(graph, betweennessLoadPath=None, show=True, savePath=None):

    G = graph
    fig, ax = plt.subplots(figsize=(10, 10))  
    ax.set_aspect('equal')

    if betweennessLoadPath:
        with open(betweennessLoadPath, 'rb') as f:
            roadBetweenness = pickle.load(f)
    else:
        roadBetweenness = calculateRoadBetweennessCentrality(G)

    edgeList = list(G.edges())
    edgeBetweenness = [(edge, roadBetweenness[G.edges[edge]['roadNumber']]) for edge in edgeList]
    edgeBetweenness.sort(key=lambda x: x[1])  # Sort edges by betweenness

    lines = []
    colours = []
    linewidths = []

    for edge, betweenness in edgeBetweenness:
        x1 = G.nodes[edge[0]]['pos'][0]
        y1 = G.nodes[edge[0]]['pos'][1]
        x2 = G.nodes[edge[1]]['pos'][0]
        y2 = G.nodes[edge[1]]['pos'][1]
        lines.append([(x1, y1), (x2, y2)])
        colours.append(betweenness)
        linewidths.append(5)# ** betweenness)

    lc = LineCollection(lines, cmap=customCmap, linewidths=linewidths)
    lc.set_array(np.array(colours))
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.axis('off')  # Turn on the axes
    plt.grid(False)  # Add a grid

    if show:
        plt.show()
    if savePath:
        plt.savefig(savePath, bbox_inches='tight')


def plotRoadsByClusteredBetweennessCentrality(graph, betweennessLoadPath=None, savePath=None, show=True, baseWidth = 3, lineWidthType = "Constant"):

    G = graph
    fig, ax = plt.subplots(figsize=(7, 12))  
    ax.set_aspect('equal')

    if betweennessLoadPath:
        with open(betweennessLoadPath, 'rb') as f:
            roadBetweenness = pickle.load(f)
    else:
        roadBetweenness = calculateRoadBetweennessCentrality(G)

    edgeList = list(G.edges())
    lines = []
    clusters = headTailBreaks(roadBetweenness)
    clusterList = list(clusters)
    colours = []
    linewidths = []
    for i in range(len(edgeList)):
        edge = edgeList[i]
        x1 = G.nodes[edge[0]]['pos'][0]
        y1 = G.nodes[edge[0]]['pos'][1]
        x2 = G.nodes[edge[1]]['pos'][0]
        y2 = G.nodes[edge[1]]['pos'][1]
        lines.append([(x1, y1), (x2, y2)])
        roadNumber = G.edges[edge]['roadNumber']
        for j in range(len(clusterList)):
            cluster = clusterList[j]
            if roadNumber in cluster:
                colours.append(j)
                if lineWidthType == "Constant":
                    linewidths.append(baseWidth)
                elif lineWidthType == "Exponential":
                    linewidths.append(baseWidth ** roadBetweenness[roadNumber])

    # Create a list of tuples where each tuple contains an edge, its color, its linewidth, and its road betweenness centrality
    edges_with_attributes = list(zip(lines, colours, linewidths, [roadBetweenness[G.edges[edge]['roadNumber']] for edge in edgeList]))

    # Sort the list of tuples based on the road betweenness centrality (in descending order)
    edges_with_attributes.sort(key=lambda x: x[3], reverse=True)

    # Unzip the sorted list of tuples back into separate lists
    lines, colours, linewidths, _ = zip(*edges_with_attributes)

    # Reverse the lists
    lines = lines[::-1]
    colours = colours[::-1]
    linewidths = linewidths[::-1]

    lc = LineCollection(lines, cmap=customCmap, linewidths=linewidths)
    lc.set_array(np.array(colours))
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.axis('off')  # Turn on the axes
    plt.grid(False)  # Add a grid


    if show:
        plt.show()
    if savePath:
        plt.savefig(savePath, bbox_inches='tight')

def saveColourbar():
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(8.5, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = plt.Normalize(0, 10)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=customCmap, norm=norm, orientation='horizontal')
    cb1.ax.tick_params(labelsize=28)
    cb1.set_ticks([0, 10])
    cb1.set_ticklabels(['Low road betweenness', 'High road betweenness'])

    plt.savefig("statsAndFigs/figs/heatmaps/colourbar2.jpeg", bbox_inches='tight',dpi=800)

def createNetworkForBoundaryComparison(show=True):
    #create a network with randomly places nodes, and no edges

    #set the np seed to ensure reproducibility
    np.random.seed(2)

    G = nx.Graph()
    for i in range(80):
        G.add_node(i, pos=(150*np.random.rand(), 150*np.random.rand()))

    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos, node_size=10, ax=ax)
        plt.axis('off')
        plt.show()

    return G

def plotBoundaryComparisonConvexHull(alpha):
    #plot the alpha shape of the network, with alpha set to infinity
    G = createNetworkForBoundaryComparison(show=False)
    plotCityBlackWithAlphaShape(G, alpha=alpha, show=True,showNodes=True)
    
