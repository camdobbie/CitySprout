from grammars import grammars
from cityGenerator import CityGenerator
import metrics
import generateMaps as maps

cityGen = CityGenerator()

"""
# Plotting and saving a city:
city1 = cityGen.generateCity(50, grammars.Organic, seed = 38, intersectRadius=0.8, plotType = "Map")
cityGen.saveCity(city1, "exampleCity.pkl")
"""



"""
# Loading and plotting alpha shape:
city1 = cityGen.loadCity("exampleCity.pkl")
maps.plotCityBlackWithAlphaShape(city1, alpha = 20)
"""



"""
# Pickling some shortest path data and road betweenness data:
city1 = cityGen.loadCity("exampleCity.pkl")
shortestPathsSavePath = "shortestPaths.pkl"
betweennessSavePath = "betweenness.pkl"
metrics.calculateRoadBetweennessCentrality(city1, shortestPathsSavePath, betweennessSavePath)
"""



"""
# Plotting a road betweenness centrality heatmap:
city1 = cityGen.loadCity("exampleCity.pkl")
betweennessLoadPath = "betweenness.pkl"
maps.plotRoadsByClusteredBetweennessCentrality(city1, betweennessLoadPath, baseWidth=2)
"""



"""
# Loading and calculating some metrics
city1 = cityGen.loadCity("exampleCity.pkl")
betweennessLoadPath = "betweenness.pkl"
shortestPathsLoadPath = "shortestPaths.pkl"
roadDensity = metrics.calculateRoadDensity(city1)
meanLivingMetric = metrics.calcMeanLivingMetric(city1, par=5, betweennessLoadPath=betweennessLoadPath)
meanCircuity = metrics.calculateAverageCircuity(city1,shortestPathsLoadPath)
print("Road density: ", roadDensity)
print("Mean living metric: ", meanLivingMetric)
print("Mean circuity: ", meanCircuity)
"""
