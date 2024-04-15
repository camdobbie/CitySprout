from grammars import grammars
from cityGenerator import CityGenerator
import metrics
import generateMaps as maps
import pickle
import matplotlib.pyplot as plt

# The organic grammar is the most similar to real cities, so we will use this to calculate the correct person per square kilometre

# As of mid 2021, London had a population of 8.797 million
# https://data.london.gov.uk/dataset/londons-population

#London has an area of 1,572 square km
#https://www.britannica.com/place/Greater-London

#Therefore, London has a population density of 5,596 people per square kilometre.
# This is the population density we will aim for in organic cities 

# run for 50 iterations the organic one, assign correct person per m road 

mode = "plotHeatmapComparison" 

cityGen = CityGenerator()

if mode == "generate":

    maxIterations =     100000000
    #grammar =           "Line" #For line, use a maxWidth of 16
    #seed =              0
    #population =        100000
    maxHeight =         None # 10km

    """
    grammarList = ["Grid","Hex","Line","Organic"]
    seedList = [0,1,2,3,4]
    populationList = [100000, 500000, 1000000, 5000000]
    """

    grammarList = ["Organic", "Line", "Hex", "Grid"]
    seedList = [0,1,2,3,4]
    populationList = [5000000]

    for population in populationList:
        for grammar in grammarList:
            grammarDict = getattr(grammars, grammar)
            if grammar == "Line":
                maxWidth = 16
            else:
                maxWidth = None

            for seed in seedList:
                fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"

                if population == 5000000:
                    complexityPath = None#f"statsAndFigs/{grammar}/{grammar}Population{population}seed{seed}Complexity.txt"
                else:
                    complexityPath = None

                print(f"Generating city with grammar: {grammar} and seed: {seed} and population: {population}")
                G = cityGen.generateCity(maxIterations, grammarDict, seed=seed, intersectRadius=0.8,  plotType=None, maxWidth=maxWidth, maxHeight=maxHeight, population=population, complexityPath=complexityPath)
                cityGen.saveCity(G, fileName)
                cityGen.clearGraph()

elif mode == "saveBlackCities":
    seed = 0
    population = 500000
    grammarList = ["Line"]

    for grammar in grammarList:

        fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
        G = cityGen.loadCity(fileName)
        #save the figure
        figName = f"statsAndFigs/figs/cityPlots/{grammar}Population{population}seed{seed}Black.pdf"
        metrics.plotCityBlack(G,show=False,savePath=figName)


elif mode == "pickleShortestPathsAndBetweenness":
    seedList = [0,1,2,3,4]
    grammarList = ["Organic","Line","Hex","Grid"]
    population = 1000000

    for seed in seedList:
        for grammar in grammarList:
            fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
            G = cityGen.loadCity(fileName)
            shortestPathsSavePath = f"statsAndFigs/shortestPathsData/{grammar}/{grammar}Population{population}seed{seed}ShortestPaths.pkl"
            betweennessSavePath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}Betweenness.pkl"
            print(f"Calculating and shelving shortest paths and betweenness for {grammar} city (seed {seed}) with population {population}")
            metrics.calculateRoadBetweennessCentrality(G, shortestPathsSavePath, betweennessSavePath)

elif mode == "calculateAverageCircuity":
    grammarList = ["Line"]
    population = 1000000
    seedList = [5,8]
    for grammar in grammarList:
        for seed in seedList:
            cityPath = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
            #G = cityGen.loadCity(cityPath)

            G = cityGen.generateCity(100000000, getattr(grammars, grammar), seed=seed, intersectRadius=0.8,  plotType=None, maxWidth=16, maxHeight=None, population=population, complexityPath=None)
            cityGen.saveCity(G, cityPath)
            averageCircuity = metrics.calculateAverageCircuity(G)
            print(f"Average circuity for {grammar} city (seed {seed}) with population {population}: {averageCircuity}")
            #add data to a new line in text file at statsAndFigs/averageCircuity.txt
            with open("statsAndFigs/averageCircuity.txt", "a") as f:
                f.write(f"{grammar},{population},{seed},{averageCircuity}\n")
            cityGen.clearGraph()

elif mode == "plotBetweenness":
    grammar = "Organic" 
    population = 1000000
    seed = 2

    fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
    G = cityGen.loadCity(fileName)
    betweennessPath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}Betweenness.pkl"
    maps.plotRoadsByClusteredBetweennessCentrality(G,betweennessLoadPath=betweennessPath,show=False,baseWidth=4,lineWidthType="Constant")
    #zoom in on a section of the city, between x = -2 and x = 2, y = -2 and y = 2
    plt.xlim(-30,30)
    plt.ylim(-45,45)

    #plt.savefig(f"statsAndFigs/figs/heatmaps/70x120Clustered{grammar}Population{population}seed{seed}V2.pdf",bbox_inches='tight')
    plt.savefig(f"statsAndFigs/figs/heatmaps/70x120Clustered{grammar}Population{population}seed{seed}V2.jpg",bbox_inches='tight',dpi=800)

elif mode == "plotHeatmapComparison":
    #G = cityGen.loadCity("savedCities/Grid/100000/GridPopulation100000seed0.pkl")

    #maps.plotRoadsByClusteredBetweennessCentrality(G,savePath="statsAndFigs/figs/heatmaps/ClusteredGrid100000heatmapV2.pdf",show=False,baseWidth=5)
    #maps.plotRoadsByBetweennessCentrality(G,savePath="statsAndFigs/figs/heatmaps/UnclusteredGrid100000heatmapV2.pdf",show=False)
    maps.saveColourbar()

elif mode == "calculatePopulationDensity":
    grammarList = ["Organic","Line","Hex","Grid"]
    seedList = [0,1,2,3,4,5,6,7,8,9] 
    population = 1000000
    for grammar in grammarList:
        for seed in seedList:
            fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
            G = cityGen.loadCity(fileName)
            populationDensity = metrics.calculateAlphaShapePopulationDensity(G)
            with open("statsAndFigs/populationDensity.txt", "a") as f:
                f.write(f"{grammar},{population},{seed},{populationDensity}\n")

elif mode == "calculateRoadDensity":
    grammarList = ["Organic","Line","Hex","Grid"]
    seedList = [0,1,2,3,4,5,6,7,8,9] 
    population = 1000000
    for grammar in grammarList:
        for seed in seedList:
            fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
            G = cityGen.loadCity(fileName)
            roadLengthPerUnitArea = metrics.calculateRoadDensity(G)
            with open("statsAndFigs/roadDensity.txt", "a") as f:
                f.write(f"{grammar},{population},{seed},{roadLengthPerUnitArea}\n")


elif mode == "pickleBetweenness":
    grammarList = ["Line","Grid","Hex"]
    seedList = [5,6,7,8,9]
    population = 1000000
    for grammar in grammarList:
        for seed in seedList:
            #in the specific case of seed 5, organic, just move on
            if grammar == "Hex":
                fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}Fixed.pkl"
                betweennessSavePath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}FixedBetweenness.pkl"
            else:
                fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
                betweennessSavePath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}Betweenness.pkl"
            G = cityGen.loadCity(fileName)
            print(f"Calculating and pickling betweenness for {grammar} city (seed {seed}) with population {population}")
            metrics.calculateRoadBetweennessCentrality(G, shortestPathsSavePath=None, betweennessSavePath=betweennessSavePath)

# average circuity for line 5 and 8
# calc population density for 10 cities each grammar and write to file
# pickle betweenness for seeds 1-4 for each grammar (0 is already done)
            
elif mode == "plotDevelopment":
    seed = 8
    iterationList = [0,1,3,5,8]
    for iterNo in iterationList:
        G = cityGen.generateCity(iterNo, grammars.Organic, seed=seed, intersectRadius=0.8, plotType = None)
        maps.plotCityBlack(G,showNodes=True,figsize=(2,4),xlim=(-7.5,7.5),ylim=(-16.5,16.5),show=False,savePath=f"statsAndFigs/figs/cityPlots/development/seed{seed}iterations{iterNo}.pdf")
        cityGen.clearGraph()

elif mode == "fixHex1000000":
    grammar = "Hex"
    population = 1000000
    seedList = [5,6,7,8,9]
    roadNumberChanges = {
        5: {35102: 35099, 35103: 35100, 35104: 35101},
        6: {38086: 38083, 38087: 38084, 38088: 38085},
        7: {41144: 41141, 41145: 41142, 41146: 41143},
        8: {44286: 44283, 44287: 44284, 44288: 44285},
        9: {47347: 47344, 47348: 47345, 47349: 47346}
    }
    cityGen.clearGraph()
    for seed in seedList:
        fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
        G = cityGen.loadCity(fileName)

        # Change road numbers
        for u, v, data in G.edges(data=True):
            if data['roadNumber'] in roadNumberChanges[seed]:
                data['roadNumber'] = roadNumberChanges[seed][data['roadNumber']]

        # Save the fixed city
        fixedFileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}Fixed.pkl"
        cityGen.saveCity(G, fixedFileName)

elif mode == "pickleBetweennessFixedHex":
    grammar = "Hex"
    population = 1000000
    seedList = [0,1,2,3,4]
    for seed in seedList:
        fixedFileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}Fixed.pkl"
        G = cityGen.loadCity(fixedFileName)

        shortestPathsSavePath = f"statsAndFigs/shortestPathsData/{grammar}/{grammar}Population{population}seed{seed}FixedShortestPaths.pkl"
        betweennessSavePath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}FixedBetweenness.pkl"
        print(f"Calculating and pickling betweenness for {grammar} city (seed {seed}) with population {population}")
        metrics.calculateRoadBetweennessCentrality(G, shortestPathsSavePath=shortestPathsSavePath, betweennessSavePath=betweennessSavePath)




#G = cityGen.loadCity("savedCities/Organic/1000000/OrganicPopulation1000000seed9.pkl")
#maps.plotCityBlackWithAlphaShape(G, figSize=(7,12), roadWidthType="Constant", roadWidth = 2/3, boundaryWidth=2/3,savePath="statsAndFigs/figs/cityPlots/OrganicPopulation1000000seed9.pdf",show=False)

elif mode == "saveAnimation":
    #cityGen.generateCity(80, grammars.Grid, seed=0, intersectRadius=0.8,  plotType="Map", plotColour="Black")
    #cityGen.generateCity(80, grammars.Grid, seed=0, intersectRadius=0.8,  plotType="Animation", plotColour="Black",figSize=(5,5),animationSavePath="Gridseed0iterations80.gif", plotXLimits=[-85,85], plotYLimits=[-85,85])
    
    #cityGen.generateCity(80, grammars.Organic, seed=0, intersectRadius=0.8,  plotType="Animation", plotColour="Black",figSize=(4,5),animationSavePath="Organicseed0iterations80.gif", plotXLimits=[-105,115], plotYLimits=[-145,130])

    #cityGen.generateCity(80, grammars.Hex, seed=0, intersectRadius=0.8,  plotType="Map", plotColour="Black")
    #cityGen.generateCity(80, grammars.Hex, seed=0, intersectRadius=0.8,  plotType="Animation", plotColour="Black",figSize=(5,5),animationSavePath="Hexseed0iterations80.gif", plotXLimits=[-85,85], plotYLimits=[-85,85])

    #cityGen.generateCity(100, grammars.Line, seed=0, intersectRadius=0.8,  plotType="Map", plotColour="Black", maxWidth=16)
    cityGen.generateCity(100, grammars.Line, seed=0, intersectRadius=0.8,  plotType="Animation", plotColour="Black",figSize=(1,12),animationSavePath="Lineseed0iterations100.gif", plotXLimits=[-10,10], plotYLimits=[-110,110], maxWidth=16)

# organic seed 0: figSize=(4,5),animationSavePath="Organicseed0iterations80.gif", plotXLimits=[-105,115], plotYLimits=[-145,130])
# grid seed 0: 

elif mode == "saveGrammarCloseUp":
    seed = 0
    grammarList = ["Organic","Line","Hex","Grid"]
    population = 1000000
    for grammar in grammarList:
        fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
        G = cityGen.loadCity(fileName)
        maps.plotCityBlack(G,show=False)
        plt.xlim(-10,10)
        plt.ylim(-30,30)
        #plt.show()
        #save fig as a svg with tight bounding box
        plt.savefig(f"statsAndFigs/figs/cityPlots/{grammar}Population{population}seed{seed}CloseUp.svg",bbox_inches='tight')
        cityGen.clearGraph()

elif mode == "plotAlphaShapeJPEGs":
    grammar = "Organic"
    population = 1000000
    seed = 0
    alphaList = [100, 20, 2]
    fileName = f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl"
    G = cityGen.loadCity(fileName)
    for alpha in alphaList:
        maps.plotCityBlackWithAlphaShape(G, figSize=(7,12), roadWidthType="Constant", roadWidth = 2, boundaryWidth=4,savePath=f"statsAndFigs/figs/cityPlots/{grammar}Population{population}seed{seed}Alpha{alpha}2.jpg",show=False,alpha=alpha)
    cityGen.clearGraph()

elif mode == "calcLivingMetric":
    seedList = [0,1,2,3,4,5,6,7,8,9]
    population = 1000000
    grammarList = ["Organic","Line","Grid","Hex"]
    for grammar in grammarList:
        for seed in seedList:
            if grammar == "Hex":
                betweennessLoadPath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}FixedBetweenness.pkl"
                G = cityGen.loadCity(f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}Fixed.pkl")
            else:
                betweennessLoadPath = f"statsAndFigs/betweennessData/{grammar}/{grammar}Population{population}seed{seed}Betweenness.pkl"
                G = cityGen.loadCity(f"savedCities/{grammar}/{population}/{grammar}Population{population}seed{seed}.pkl")
            meanLivingValue = metrics.calcMeanLivingMetric(G, par=5, betweennessLoadPath=betweennessLoadPath)
            with open("statsAndFigs/livingMetric5.txt", "a") as f:
                f.write(f"{grammar},{population},{seed},{meanLivingValue}\n")

# FOR FIXING HEX:
    # 100,000 population
    # seed 0: 4 becomes 1, 5 becomes 2, 6 becomes 3
    # seed 1: 198 becomes 195, 199 becomes 196, 200 becomes 197
    # seed 2: 419 becomes 416, 420 becomes 417, 421 becomes 418
    # seed 3: 648 becomes 645, 649 becomes 646, 650 becomes 647
    # seed 4: 884 becomes 881, 885 becomes 882, 886 becomes 883
        
    # 1,000,000 population
    # seed 0: 35022 becomes 35019, 35023 becomes 35020, 35024 becomes 35021
    # seed 1: 38012 becomes 38009, 38013 becomes 38010, 38014 becomes 38011
    # seed 2: 41031 becomes 41028, 41032 becomes 41029, 41033 becomes 41030
    # seed 3: 44059 becomes 44056, 44060 becomes 44057, 44061 becomes 44058
    # seed 4: 47062 becomes 47059, 47063 becomes 47060, 47064 becomes 47061

    # seed 5: 35102 becomes 35099, 35103 becomes 35100, 35104 becomes 35101
    # seed 6: 38086 becomes 38083, 38087 becomes 38084, 38088 becomes 38085
    # seed 7: 41144 becomes 41141, 41145 becomes 41142, 41146 becomes 41143
    # seed 8: 44286 becomes 44283, 44287 becomes 44284, 44288 becomes 44285
    # seed 9: 47347 becomes 47344, 47348 becomes 47345, 47349 becomes 47346
