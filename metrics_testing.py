from grammars import grammars
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import kurtosis
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
import random
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from shapely.geometry import GeometryCollection, Polygon, MultiLineString
from shapely.ops import unary_union, polygonize
from shapely import geometry
import matplotlib.patches as patches
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from tqdm import tqdm
import pickle
import shelve
import numpy as np
from scipy.stats import kendalltau

def calculateShortestPaths(G, shortestPathsSavePath=None):
    # calculate the length of each edge and add it as a 'weight' attribute
    for u, v in G.edges():
        pos_u = np.array(G.nodes[u]['pos'])
        pos_v = np.array(G.nodes[v]['pos'])
        length = np.linalg.norm(pos_u - pos_v)
        G[u][v]['length'] = length

    # calculate the shortest path between each pair of nodes in the graph
    shortestPaths = {}
    shortestPathsLengths = {}
    for source in tqdm(G.nodes(), desc="Calculating shortest paths"):
        shortestPaths[source] = nx.single_source_dijkstra_path(G, source, weight='length')
        # calculate the length of the shortest path
        for target, path in shortestPaths[source].items():
            shortestPathsLengths[(source, target)] = sum(G[u][v]['length'] for u, v in zip(path[:-1], path[1:]))

    if shortestPathsSavePath:
        with open(shortestPathsSavePath, 'wb') as f:
            pickle.dump((shortestPaths, shortestPathsLengths), f)

    return shortestPaths, shortestPathsLengths

def pathToRoads(G, path):
    """Convert a node path to a sequence of roadNumbers along the path, 
    avoiding repeated consecutive roads."""
    roads = []
    prev_road = None
    for u, v in zip(path[:-1], path[1:]):
        road = G.edges[u, v]['roadNumber']
        if road != prev_road:
            roads.append(road)
            prev_road = road
    return roads

def pathToEdges(path):
    return [(u, v) for u, v in zip(path[:-1], path[1:])]

def getRoadRanks(roadBetweenness):
    sorted_roads = sorted(roadBetweenness.items(),
                           key=lambda x: x[1])
    return {road: rank for rank, (road, _) in enumerate(sorted_roads)}

def roadBetweennessSequence(roadSeq, roadBetweenness):
    """Return the betweenness values along a road sequence."""
    return [roadBetweenness[road] for road in roadSeq]

def pathTauRoads(roadSeq, roadBetweenness):
    seq = [roadBetweenness[r] for r in roadSeq]
    if len(seq) < 2:
        return np.nan
    tau, _ = kendalltau(seq, range(len(seq)))
    return tau

def pathTauEdges(edgeSeq, edgeBetweenness):
    seq = [edgeBetweenness[e] for e in edgeSeq]
    if len(seq) < 2:
        return np.nan

    tau, _ = kendalltau(-np.array(seq), range(len(seq)))
    return tau

def calculateRawEdgeBetweenness(shortestPaths):

    # create a dictionary of edges and the number of shortest paths that pass through each edge
    edgeBetweenness = {} 

    for i, source in enumerate(tqdm(shortestPaths, desc="Processing shortest paths for edge betweenness"), start=1):
        for target in shortestPaths[source]:
            # if the source and target are different nodes
            if source != target:
                # get the shortest path between the source and target nodes
                path = shortestPaths[source][target]
                # iterate through each edge in the path
                for i in range(len(path)-1):
                    # get the edge
                    edge = (path[i], path[i+1])
                    # if the edge is not in the dictionary, add it
                    if edge not in edgeBetweenness:
                        edgeBetweenness[edge] = 1
                    # if the edge is in the dictionary, increment the value
                    else:
                        edgeBetweenness[edge] += 1

    return edgeBetweenness

# def calculateLivingMetricKendallRoads(G, shortestPaths, rawRoadBetweenness):

#     centrePaths = shortestPaths[0]

#     taus = []
#     weights = []

#     for target, path in tqdm(centrePaths.items(), desc="Calculating LivingMetricV2 by road"):
#         if target == 0:
#             continue

#         roadSeq = pathToRoads(G, path)
#         k = len(roadSeq)

#         if k < 2:
#             continue

#         tau = pathTauRoads(roadSeq, rawRoadBetweenness)

#         if not np.isnan(tau):
#             taus.append(tau)
#             weights.append(k)  # weight by number of roads

#     livingMetricV2 = np.average(taus, weights=weights)
#     return livingMetricV2

# def calculateLivingMetricKendallEdges(shortestPaths, rawEdgeBetweenness):

#     centrePaths = shortestPaths[0]

#     taus = []
#     weights = []

#     for target, path in tqdm(centrePaths.items(),
#                              desc="Calculating LivingMetricV2 (edge)"):
#         if target == 0:
#             continue

#         edgeSeq = pathToEdges(path)
#         k = len(edgeSeq)

#         if k < 2:
#             continue

#         tau = pathTauEdges(edgeSeq, rawEdgeBetweenness)
#         if not np.isnan(tau):
#             taus.append(tau)
#             weights.append(k)   # weight by number of edges

#     return np.average(taus, weights=weights)

def calculateRawRoadBetweenness(G, rawEdgeBetweenness):

    # for each possible roadNumber, calculate how many shortest paths pass through that road
    roadNumbers = set(nx.get_edge_attributes(G, 'roadNumber').values())
    rawRoadBetweenness = {}
    for roadNumber in tqdm(roadNumbers, desc = "Processing roads for road betweenness"):
        rawRoadBetweenness[roadNumber] = 0
        for edge in rawEdgeBetweenness:
            if G.edges[edge]['roadNumber'] == roadNumber:
                rawRoadBetweenness[roadNumber] += rawEdgeBetweenness[edge]

    return rawRoadBetweenness

def normaliseRawRoadBetweenness(roadBetweenness, normalisedRoadBetweennessSavePath=None):

    # normalise the roadBetweenness values using min-max normalisation
    maxBetweenness = max(roadBetweenness.values())
    minBetweenness = min(roadBetweenness.values())
    for roadNumber in roadBetweenness:
        roadBetweenness[roadNumber] = (roadBetweenness[roadNumber] - minBetweenness) / (maxBetweenness - minBetweenness)


    if normalisedRoadBetweennessSavePath:
        with open(normalisedRoadBetweennessSavePath, 'wb') as f:
            pickle.dump(roadBetweenness, f)

    return roadBetweenness

""" def shelveShortestPathsAndBetweenness(G, shortestPathShelvePath, betweennessShelvePath):
    # calculate the length of each edge and add it as a 'weight' attribute
    for u, v in G.edges():
        pos_u = np.array(G.nodes[u]['pos'])
        pos_v = np.array(G.nodes[v]['pos'])
        length = np.linalg.norm(pos_u - pos_v)
        G[u][v]['length'] = length

    # calculate the shortest path between each pair of nodes in the graph
    shortestPathsShelve = shelve.open(shortestPathShelvePath)
    for source in tqdm(G.nodes(), desc="Calculating shortest paths"):
        shortestPathsShelve[str(source)] = nx.single_source_dijkstra_path(G, source, weight='length')

    # create a dictionary of edges and the number of shortest paths that pass through each edge
    edgeBetweenness = {} 

    for i, source in enumerate(tqdm(shortestPathsShelve.keys(), desc="Processing nodes"), start=1):
        # Load one shortest path into memory
        shortestPaths = shortestPathsShelve[str(source)]
        for target in shortestPaths:
            # if the source and target are different nodes
            if source != target:
                # get the shortest path between the source and target nodes
                path = shortestPaths[target]
                # iterate through each edge in the path
                for i in range(len(path)-1):
                    # get the edge
                    edge = (path[i], path[i+1])
                    # if the edge is not in the dictionary, add it
                    if edge not in edgeBetweenness:
                        edgeBetweenness[edge] = 1
                    # if the edge is in the dictionary, increment the value
                    else:
                        edgeBetweenness[edge] += 1

    # for each possible roadNumber, calculate how many shortest paths pass through that road
    roadNumbers = set(nx.get_edge_attributes(G, 'roadNumber').values())
    roadBetweenness = shelve.open(betweennessShelvePath)
    for roadNumber in tqdm(roadNumbers, desc="Calculating road betweenness"):
        roadBetweenness[str(roadNumber)] = 0  # Convert roadNumber to a string
        for edge in edgeBetweenness:
            if G.edges[edge]['roadNumber'] == roadNumber:
                roadBetweenness[str(roadNumber)] += edgeBetweenness[edge]  # Convert roadNumber to a string

    # normalise the roadBetweenness values using min-max normalisation
    maxBetweenness = max(roadBetweenness.values())
    minBetweenness = min(roadBetweenness.values())
    for roadNumber in tqdm(roadBetweenness, desc="Normalising betweenness"):
        roadBetweenness[roadNumber] = (roadBetweenness[roadNumber] - minBetweenness) / (maxBetweenness - minBetweenness)

    shortestPathsShelve.close()
    roadBetweenness.close()
 """

def calculateRoadConnectionDict(G):
    roadConnectionDict = {}
    for roadNumber in nx.get_edge_attributes(G, 'roadNumber').values():
        roadConnectionDict[roadNumber] = []
    for node in G.nodes():
        connectedRoads = set()
        for edge in G.edges(node, data=True):
            if 'roadNumber' in edge[2]:
                connectedRoads.add(edge[2]['roadNumber'])
        for roadNumber in connectedRoads:
            # Exclude the road itself from its list of connected roads
            roadConnectionDict[roadNumber].extend([connectedRoad for connectedRoad in connectedRoads if connectedRoad != roadNumber])
    return roadConnectionDict

def calculateLivingMetricScore(G, roadBetweenness, lambda_=10):

    roadRanks = getRoadRanks(roadBetweenness)
    roadConnections = calculateRoadConnectionDict(G)

    scores = {}

    for road, neighbours in roadConnections.items():
        r_i = roadRanks[road]

        score = 0.0
        for nbr in neighbours:
            r_j = roadRanks[nbr]
            d = abs(r_i - r_j)
            score += np.exp(-lambda_ * d)

        scores[road] = score

    return np.mean(list(scores.values()))


def calculateLivingMetricScoreNormalised(
    G, roadBetweenness, lambda_=1
):
    roadRanks = getRoadRanks(roadBetweenness)
    roadConnections = calculateRoadConnectionDict(G)

    N = len(roadRanks)
    scores = {}

    for road, neighbours in roadConnections.items():
        if not neighbours:
            continue

        r_i = roadRanks[road]
        s = 0.0

        for nbr in neighbours:
            r_j = roadRanks[nbr]
            d = abs(r_i - r_j) / (N - 1)   # normalised rank distance
            s += np.exp(-lambda_ * d)

        scores[road] = s / len(neighbours)  # degree-normalised

    return np.mean(list(scores.values()))

def calculateAverageCircuity(G,shortestPathsLoadPath=None):
    #average circuity is defined as the mean of all circuity values for each pair of nodes in the graph.
    # for each pair of nodes, circuity is the shortest path length divided by the euclidean distance between the nodes
    if shortestPathsLoadPath:
        print("Loading shortest paths")
        with open(shortestPathsLoadPath, 'rb') as f:
            shortestPaths, shortestPathsLengths = pickle.load(f)
        print("Calculating circuity")
    else:
        shortestPaths, shortestPathsLengths = calculateShortestPaths(G)
    circuityList = []
    for source in tqdm(shortestPaths, desc="Calculating circuity"):
        for target in shortestPaths[source]:
            if source != target:
                shortestPathLength = shortestPathsLengths[(source, target)]
                x1 = G.nodes[source]['pos'][0]
                y1 = G.nodes[source]['pos'][1]
                x2 = G.nodes[target]['pos'][0]
                y2 = G.nodes[target]['pos'][1]
                euclideanDistance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                circuity = shortestPathLength / euclideanDistance
                circuityList.append(circuity)
    averageCircuity = sum(circuityList) / len(circuityList)
    return averageCircuity



def singleHeadTailBreak(roadCentralityDict):
    #sort the dictionary by value in descending order
    sortedDict = dict(sorted(roadCentralityDict.items(), key=lambda item: item[1]))
    #calculate the mean value
    mean = sum(sortedDict.values())/len(sortedDict)
    #split roadCentralityDict on either side of the mean
    head = {k: v for k, v in sortedDict.items() if v >= mean}
    tail = {k: v for k, v in sortedDict.items() if v < mean}
    return tail, head

def headTailBreaks(roadCentralityDict, clusters = None, counter = 0):
    """function to perform head/tail breaks on the data, and output a list of the parts.
    the clusters parameter is used to pass the clusters list between recursive calls of the function
    the output is a list of dictionaries, each containing the roadNumbers and betweenness centrality values of the roads in that cluster"""
    if clusters is None:
        clusters = []
    #if there is only one road in the dictionary, add it to the clusters list and return the list
    if len(roadCentralityDict) == 1:
        clusters.append(roadCentralityDict)
        return clusters
    #if there are two roads in the dictionary, split the dictionary into two parts and add them to the clusters list
    elif len(roadCentralityDict) == 2:
        tail, head = singleHeadTailBreak(roadCentralityDict)
        clusters.append(tail)
        clusters.append(head)
        return clusters
    #if there are more than two roads in the dictionary, split the dictionary into two parts and call the function again on each part
    else:
        tail, head = singleHeadTailBreak(roadCentralityDict)
        clusters.append(tail)
        return headTailBreaks(head, clusters, counter+1)



def calculateTotalRoadLength(G):
    totalLength = 0
    for edge in G.edges():
        x1 = G.nodes[edge[0]]['pos'][0]
        y1 = G.nodes[edge[0]]['pos'][1]
        x2 = G.nodes[edge[1]]['pos'][0]
        y2 = G.nodes[edge[1]]['pos'][1]
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        totalLength += length
    return totalLength

def calculatePopulation(G):
    totalLength = calculateTotalRoadLength(G)
    return round(totalLength * 154.15)#67.68)

def alpha_shape(G, alpha=20):

    # Get the positions of the nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Convert the positions to a list of points
    points = [Point(p) for p in pos.values()]

    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    # calculate a,b,c, which are the side lengths of the triangles
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    # calculate s, which is the semi-perimeter of the triangles
    s = ( a + b + c ) / 2.0
    # calculate the area of the triangles
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    # calculate the circumradius of the triangles
    circums = a * b * c / (4.0 * areas)
    # filter the triangles by the circumradius
    filtered = triangles[circums < alpha]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points

def calculateAlphaShapeArea(G, alpha=20):
    polygon, edge_points = alpha_shape(G, alpha)
    area = polygon.area
    return area

def calculateAlphaShapePopulationDensity(G, alpha=20):
    population = calculatePopulation(G)
    area = calculateAlphaShapeArea(G, alpha)
    areaKm = area / 100
    print(f"Population density (calculated using the alpha shape): {round(population/areaKm)} people per square km.")
    return round(population/areaKm)

def calculateRoadDensity(G):
    totalLength = calculateTotalRoadLength(G)
    area = calculateAlphaShapeArea(G)
    print(f"Road density (calculated using the alpha shape): {totalLength/area} units of length per unit of area.")
    return totalLength/area

# def calculateRoadConnectionDict(G):
#     roadConnectionDict = {}
#     for roadNumber in nx.get_edge_attributes(G, 'roadNumber').values():
#         roadConnectionDict[roadNumber] = []
#     for node in G.nodes():
#         connectedRoads = set()
#         for edge in G.edges(node, data=True):
#             if 'roadNumber' in edge[2]:
#                 connectedRoads.add(edge[2]['roadNumber'])
#         for roadNumber in connectedRoads:
#             # Exclude the road itself from its list of connected roads
#             roadConnectionDict[roadNumber].extend([connectedRoad for connectedRoad in connectedRoads if connectedRoad != roadNumber])
#     return roadConnectionDict

# def calculateLivingMetricDict(G, par, betweennessLoadPath):
#     roadBetweenness = pickle.load(open(betweennessLoadPath, 'rb'))
#     roadConnectionDict = calculateRoadConnectionDict(G)
#     livingMetricDict = {}
#     roadNumberLookup = {v: k for k, v in roadBetweenness.items()}
#     for roadNumber in tqdm(nx.get_edge_attributes(G, 'roadNumber').values(), desc="Calculating living metric"):
#         roadBetweennessList = list(roadBetweenness.values())
#         roadBetweennessList.sort()
#         roadIndex = roadBetweennessList.index(roadBetweenness[roadNumber])
#         lowerBound = max(0, roadIndex - par)
#         upperBound = min(len(roadBetweennessList), roadIndex + par)
#         connectedRoads = 0
#         for i in range(lowerBound, upperBound):
#             otherRoadNumber = roadNumberLookup[roadBetweennessList[i]]
#             # Only increment connectedRoads if the other road is directly connected to the current road
#             if otherRoadNumber in roadConnectionDict[roadNumber]:
#                 connectedRoads += 1
#         livingMetricDict[roadNumber] = connectedRoads
#     return livingMetricDict

# def calculateDictOfLivingValues(G, par, betweennessLoadPath):
#     livingDict = calculateLivingMetricDict(G, par=par, betweennessLoadPath=betweennessLoadPath)
#     # create a dictionary of the living values, and how many times they occur
#     dictOfLivingValues = {}
#     for key, value in livingDict.items():
#         if value not in dictOfLivingValues:
#             dictOfLivingValues[value] = 1
#         else:
#             dictOfLivingValues[value] += 1

# def calcMeanLivingMetric(G,par,betweennessLoadPath):
#     livingDict = calculateLivingMetricDict(G, par=par, betweennessLoadPath=betweennessLoadPath)
#     return sum(livingDict.values())/len(livingDict)