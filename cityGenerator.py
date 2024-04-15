import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
import numpy as np
import time
import pickle
import metrics as optim
from matplotlib.animation import PillowWriter

class CityGenerator:
    def __init__(self):
        self.G = nx.Graph()
        self.roadTypes = ["m","l","s"]
        self.nodeTypesGen = ["K","T","B","L"]
#make nodeRoadsAndTypes a list of all possible combinations of roadTypes and nodeTypesGen
        self.nodeRoadsAndTypes = [roadType + nodeType for roadType in self.roadTypes for nodeType in self.nodeTypesGen]
        self.roadNumbers = [0]

    def clearGraph(self):
        self.G.clear()

    def getRandomThetaList(self, minimum, maximum, randDirection):
        theta = random.uniform(minimum, maximum)
        if randDirection:
            theta = theta*random.choice([-1, 1])
        return theta

    def getRandomThetaSingle(self, theta,randDirection):
        theta = theta
        if randDirection:
            theta = theta*random.choice([-1, 1])
        return theta

    def calcIntersectionPoint(self, p1, p2, p3, p4):
        if (p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]) == 0:
            return False
        else:
            # Calculate the intersection point
            x = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]))
            y = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0])) / ((p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1] - p2[1])*(p3[0] - p4[0]))
            # Check if the intersection point is at the endpoint of either line
            if (x == p1[0] and y == p1[1]) or (x == p2[0] and y == p2[1]) or (x == p3[0] and y == p3[1]) or (x == p4[0] and y == p4[1]):
                return False
            # check is the intersection point is on the line segments
            elif (x >= min(p1[0],p2[0]) and x <= max(p1[0],p2[0]) and y >= min(p1[1],p2[1]) and y <= max(p1[1],p2[1])) and (x >= min(p3[0],p4[0]) and x <= max(p3[0],p4[0]) and y >= min(p3[1],p4[1]) and y <= max(p3[1],p4[1])):
                return (x,y)
            else:
                return False

    def calculateNewPosition(self, node, theta, length):

        newNode = list(self.G.nodes())[-1] + 1
        currentPosition = self.G.nodes[node]['pos']  # get the position of the current node
        currentDirection = self.G.nodes[node]['incEdge'] # get the direction of the incoming edge
        currentDirectionUnit = (currentDirection[0]/(currentDirection[0]**2 + currentDirection[1]**2)**0.5, currentDirection[1]/(currentDirection[0]**2 + currentDirection[1]**2)**0.5) # Normalise current direction into a unit vector
        currentDirectionRotated = (currentDirectionUnit[0] * math.cos(theta) - currentDirectionUnit[1] * math.sin(theta), currentDirectionUnit[0] * math.sin(theta) + currentDirectionUnit[1] * math.cos(theta)) # rotate the current direction by theta
        newDirection = (currentDirectionRotated[0] * length, currentDirectionRotated[1] * length) # scale the rotated direction by length
        newPosition = (currentPosition[0] + newDirection[0], currentPosition[1] + newDirection[1]) # calculate the new position

        return newNode,newPosition
        # ...

    def checkProximity(self, position, intersectRadius):
        for existingNode in self.G.nodes():
            existingPosition = self.G.nodes[existingNode]['pos']
            distance = ((position[0] - existingPosition[0])**2 + (position[1] - existingPosition[1])**2)**0.5
            if distance < intersectRadius:
                #print(f"the node at position {position} is too close to node {existingNode} at position {existingPosition}")
                return existingNode

    def checkBoundaryBoxIntersection(self, box1, box2):
        if box1[0] > box2[1] or box1[1] < box2[0] or box1[2] > box2[3] or box1[3] < box2[2]:
            return False
        else:
            return True

    def findClosestIntersection(self, node, newPosition):
        newEdgeXMin = min(self.G.nodes[node]['pos'][0], newPosition[0])
        newEdgeXMax = max(self.G.nodes[node]['pos'][0], newPosition[0])
        newEdgeYMin = min(self.G.nodes[node]['pos'][1], newPosition[1])
        newEdgeYMax = max(self.G.nodes[node]['pos'][1], newPosition[1])
        newEdgeBoundingBox = (newEdgeXMin,newEdgeXMax,newEdgeYMin,newEdgeYMax)

        intersections = []
        currentPosition = self.G.nodes[node]['pos']

        for edge in self.G.edges():
            if self.checkBoundaryBoxIntersection(self.G.edges[edge]['boundaryBox'], newEdgeBoundingBox):
                p1 = self.G.nodes[edge[0]]['pos']
                p2 = self.G.nodes[edge[1]]['pos']
                intersection = self.calcIntersectionPoint(p1, p2, currentPosition, newPosition)
                if intersection:  # If there is an intersection
                    distance = ((intersection[0] - currentPosition[0])**2 + (intersection[1] - currentPosition[1])**2)**0.5
                    intersections.append((intersection, distance))

        if intersections:
            closestIntersection = min(intersections, key=lambda x: x[1])[0]
            return closestIntersection
        else:
            return False
            # ...

    def calcMinDistanceToType(self, position, nodeType):

        distance = math.inf

        if nodeType in self.nodeRoadsAndTypes:
            for node in self.G.nodes():
                if self.G.nodes[node]['roadType'] + self.G.nodes[node]['nodeType']== nodeType:
                    checkDistance = ((position[0] - self.G.nodes[node]['pos'][0])**2 + (position[1] - self.G.nodes[node]['pos'][1])**2)**0.5
                    if checkDistance < distance:
                        distance = checkDistance
            return distance
        else:
            return math.inf

    def createNodes(self, node, changeNodeTo, theta, length, newRoadType, newNodeType, newRoad ,weight=1, intersectRadius=None, maxWidth=None, maxHeight=None):

        newNode,newPosition = self.calculateNewPosition(node, theta, length)
        newRoadAndType = newRoadType + newNodeType;

        if maxWidth:
            if newPosition[0] > maxWidth/2:
                newNodeType = "BOUNDARY"
                newPosition = (maxWidth/2, newPosition[1])
            if newPosition[0] < -maxWidth/2:
                newNodeType = "BOUNDARY"
                newPosition = (-maxWidth/2, newPosition[1])

        if maxHeight:
            if newPosition[1] > maxHeight/2:
                newNodeType = "BOUNDARY"
                newPosition = (newPosition[0], maxHeight/2)
            if newPosition[1] < -maxHeight/2:
                newNodeType = "BOUNDARY"
                newPosition = (newPosition[0], -maxHeight/2)

        # Check if the new edge intersects with any existing edge
        closestIntersection = self.findClosestIntersection(node, newPosition)
        if closestIntersection:
            newPosition = closestIntersection

        tooCloseNode = self.checkProximity(newPosition, intersectRadius)

        self.G.nodes[node]['nodeType'] = changeNodeTo # change the node type of the current node

        newDirection = (newPosition[0] - self.G.nodes[node]['pos'][0], newPosition[1] - self.G.nodes[node]['pos'][1]) # calculate the direction of the new edge

        if tooCloseNode or tooCloseNode == 0:
        
            if node != tooCloseNode:
                boundaryBox = (min(self.G.nodes[node]['pos'][0], self.G.nodes[tooCloseNode]['pos'][0]), max(self.G.nodes[node]['pos'][0], self.G.nodes[tooCloseNode]['pos'][0]), min(self.G.nodes[node]['pos'][1], self.G.nodes[tooCloseNode]['pos'][1]), max(self.G.nodes[node]['pos'][1], self.G.nodes[tooCloseNode]['pos'][1]))

                if newRoad:
                    newRoadNumber = self.roadNumbers[-1] + 1
                else:
                    newRoadNumber = self.G.nodes[node]['incRoadNumber']


                self.G.add_edge(node, tooCloseNode,weight=weight, boundaryBox=boundaryBox, roadNumber=newRoadNumber) # add the new edge to the graph
                if newRoadNumber not in self.roadNumbers:
                    self.roadNumbers.append(newRoadNumber)
        else:

            self.G.add_node(newNode, nodeType=newNodeType, pos=newPosition, incEdge=newDirection, roadType = newRoadType)
            boundaryBox = (min(self.G.nodes[node]['pos'][0], newPosition[0]), max(self.G.nodes[node]['pos'][0], newPosition[0]), min(self.G.nodes[node]['pos'][1], newPosition[1]), max(self.G.nodes[node]['pos'][1], newPosition[1]))

            if newRoad:
                newRoadNumber = self.roadNumbers[-1] + 1
            else:
                newRoadNumber = self.G.nodes[node]['incRoadNumber']

            self.G.add_edge(node, newNode,weight=weight, boundaryBox=boundaryBox, roadNumber=newRoadNumber) # add the new edge to the graph
            #add the incRoadNumber attribute to the new node
            self.G.nodes[newNode]['incRoadNumber'] = newRoadNumber
            #if newRoadNumber is not in the list of roadNumbers, add it
            if newRoadNumber not in self.roadNumbers:
                self.roadNumbers.append(newRoadNumber)

    def generateCity(self, iterations, grammar, seed=42, intersectRadius=0.8, maxWidth = None, maxHeight = None, plotType="Map", showNodes=False,
                     nodeLabelType=None, edgeLabelType=None, complexityPath=None, maxRoadSegments = None, maxRoadLength=None, population = None, plotColour = "Default", animationSavePath=None, plotXLimits=None, plotYLimits=None, figSize=None):

        random.seed(seed)
        np.random.seed(seed)

        startTime = time.time()

        self.G.add_node(0, nodeType='Start', roadType="m", pos=(0,0), incEdge=(0,1),incRoadNumber=0) # incEdge is the direction of the incoming edge
        
        if plotType == "Animation" or plotType == "Map":
            if figSize:
                fig, ax = plt.subplots(figsize=figSize)
            else:
                fig, ax = plt.subplots()
            ax.set_aspect('equal')

        def graphSettings():
            if showNodes:
                node_size = 10
            else:
                node_size = 0
            if nodeLabelType == "Node Type":
                with_labels = True
                labels=nx.get_node_attributes(self.G, 'nodeType')
            elif nodeLabelType == "Node Number":
                with_labels = True
                labels = {node: node for node in self.G.nodes()}
            elif nodeLabelType == "Road Type":
                with_labels = True
                labels=nx.get_node_attributes(self.G, 'roadType')
            else:
                with_labels = False
                labels=None

            edges = self.G.edges()
            # Create a list of colors based on the weights of the edges
            edge_colors = ['#ffd747' if self.G[u][v]['weight'] == 1 else '#e3e3e3' if self.G[u][v]['weight'] == 0.5 else '#dbdbdb' for u, v in edges]
            edge_widths = [4 if self.G[u][v]['weight'] == 1 else 3 if self.G[u][v]['weight'] == 0.5 else 2 for u, v in edges]
            edge_widthsBlack = [x + 1 for x in edge_widths]

            pos = nx.get_node_attributes(self.G, 'pos')
            nx.draw_networkx_edges(self.G, pos, edge_color='black', width=edge_widthsBlack, ax=ax)  # Draw edges with outline
            nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)  # Draw edges
            if edgeLabelType == "Edge Weight":
                edge_labels = nx.get_edge_attributes(self.G, 'weight')
                nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=ax)

            if edgeLabelType == "Road Number":
                edge_labels = nx.get_edge_attributes(self.G, 'roadNumber')
                nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=ax)

            nx.draw_networkx_nodes(self.G, pos, node_size=node_size, ax=ax)

            if with_labels:
                nx.draw_networkx_labels(self.G, pos, labels=labels, ax=ax)
            plt.axis('on')  # Turn on the axes
            plt.grid(False)  # Add a grid

        def graphSettingsBlack():
            if showNodes:
                node_size = 10
            else:
                node_size = 0
            if nodeLabelType == "Node Type":
                with_labels = True
                labels=nx.get_node_attributes(self.G, 'nodeType')
            elif nodeLabelType == "Node Number":
                with_labels = True
                labels = {node: node for node in self.G.nodes()}
            elif nodeLabelType == "Road Type":
                with_labels = True
                labels=nx.get_node_attributes(self.G, 'roadType')
            else:
                with_labels = False
                labels=None

            edges = self.G.edges()
            edge_widths = [1.5 if self.G[u][v]['weight'] == 1 else 1 if self.G[u][v]['weight'] == 0.5 else 0.5 for u, v in edges]

            pos = nx.get_node_attributes(self.G, 'pos')
            nx.draw_networkx_edges(self.G, pos, edge_color='black', width=edge_widths, ax=ax)

            for spine in ax.spines.values():
                spine.set_visible(False)

        def applyLSystem():

            #nodes = list(self.G.nodes())  # Create a copy of the nodes
                
            # create a list of the nodes in the graph which have nodeTypes which are not "K" or "BOUNDARY"
            nodes = [node for node in self.G.nodes() if self.G.nodes[node]['nodeType'] not in ["K","BOUNDARY"]]

            for node in nodes:

                nodeRoadAndType = self.G.nodes[node]['roadType'] + self.G.nodes[node]['nodeType']
                #choose one of the rules based on occurProb

                #if there is no rule for nodes of this type, skip this node
                if nodeRoadAndType not in grammar:
                    continue
                rules = grammar[nodeRoadAndType]
                occurProbs = [rule['occurProb'] for rule in rules]
                #if the sum of the occurProbs is not 1, normalize them
                if sum(occurProbs) != 1:
                    occurProbs = [prob/sum(occurProbs) for prob in occurProbs]
                #create a list of numbers from 0 to the length of the rules list
                ruleIndices = list(range(len(rules)))
                #randomly select one of the rules based on the occurProbs
                idealRuleIndex = np.random.choice(ruleIndices, p=occurProbs)
                #create a random permutation of ruleIndices, but ensure that the idealRuleIndex is in the first position
                ruleIndices.remove(idealRuleIndex)
                random.shuffle(ruleIndices)
                ruleIndices.insert(0,idealRuleIndex)

                """
                for ruleIndex in ruleIndices:
                    rule = rules[ruleIndex]
                    #check if the rule can be applied by checking the minDistances
                    if not all([G.nodes[node]['minDistances'][nodeType] >= minDistance for nodeType,minDistance in rule['minDistances'].items()]):
                        continue #if the rule cannot be applied, skip it
                    else:
                        break #if the rule can be applied, break out of the loop and apply it
                """

                #check if the rule can be applied, by calculating the minDistances to all nodes of the types in the rule
                for ruleIndex in ruleIndices:
                    rule = rules[ruleIndex]
                    minDistances = {}
                    for nodeType in rule['minDistances']:
                        minDistances[nodeType] = self.calcMinDistanceToType(self.G.nodes[node]['pos'], nodeType)
                    if not all([minDistances[nodeType] >= minDistance for nodeType,minDistance in rule['minDistances'].items()]):
                        continue
                    else:
                        break


                if not len(rule['thetas']) == len(rule['lengths']) == len(rule['newRoadTypes']) == len(rule['newNodeTypes']) == len(rule['newRoad']) == len(rule['randDirection']):
                    raise Exception("The number of thetas, lengths, newRoadTypes and newNodeTypes must be the same")
                for i in range(len(rule['thetas'])):
                    if type(rule['thetas'][i]) == list:
                        theta = self.getRandomThetaList(rule['thetas'][i][0],rule['thetas'][i][1],rule['randDirection'][i])
                    else:
                        theta = self.getRandomThetaSingle(rule['thetas'][i],rule['randDirection'][i])
                    
                    if rule['newRoadTypes'][i]=="m":
                        weight = 1
                    elif rule['newRoadTypes'][i]=="l":
                        weight = 0.5
                    elif rule['newRoadTypes'][i]=="s":
                        weight = 0.25

                    self.createNodes(node, rule['changeNodeTo'], theta, rule['lengths'][i], rule['newRoadTypes'][i], rule['newNodeTypes'][i], rule['newRoad'][i], weight=weight, intersectRadius=intersectRadius, maxWidth=maxWidth, maxHeight=maxHeight)

                    if maxRoadSegments is not None and len(self.G.edges()) > maxRoadSegments:
                        return
                    
                    if maxRoadLength is not None and optim.calculateTotalRoadLength(self.G) > maxRoadLength:
                        return
                    
                    if population is not None and optim.calculatePopulation(self.G) > population:
                        return
                
            if complexityPath:

                numNodes = len(self.G.nodes())

                endTime = time.time()
                timeTaken = endTime - startTime

                with open(complexityPath, "a") as file:
                    file.write(f"{numNodes},{timeTaken}\n")


        def update(i):
            print(f"Starting iteration: {i+1}/{iterations}")

            if maxRoadSegments:
                print(f"Road segments: {self.G.number_of_edges()}")
            if maxRoadLength:
                print(f"Road length: {optim.calculateTotalRoadLength(self.G)}")
            if population:
                print(f"Population: {optim.calculatePopulation(self.G)}")

            ax.clear()

            applyLSystem()

            if maxRoadSegments is not None and self.G.number_of_edges() >= maxRoadSegments:
                return
            
            if maxRoadLength is not None and optim.calculateTotalRoadLength(self.G) > maxRoadLength:
                return
            
            if population is not None and optim.calculatePopulation(self.G) > population:
                return

            if plotColour == "Default":
                graphSettings()
            elif plotColour == "Black":
                graphSettingsBlack()

            if plotXLimits:
                ax.set_xlim(plotXLimits)
            if plotYLimits:
                ax.set_ylim(plotYLimits)
            
            
            """
            # Add a text box displaying the current iteration
            iteration_text = f"Iteration: {i+1}/{iterations}"
            ax.text(0.02, 0.95, iteration_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            """

        if plotType == "Animation":
                    # Add these lines to remove the axes and border before the animation starts
            ax.axis('off')
            ani = animation.FuncAnimation(fig, update, frames=iterations, repeat=False)
            if animationSavePath:
                ani.save(animationSavePath, writer=PillowWriter(fps=7))
                #return the graph object and stop the function
                return self.G
            plt.show()
        else:
            
            for i in range(iterations):
                print(f"Starting iteration: {i+1}/{iterations}")

                if maxRoadSegments:
                    print(f"Road segments: {self.G.number_of_edges()}")
                if maxRoadLength:
                    print(f"Road length: {optim.calculateTotalRoadLength(self.G)}")
                if population:
                    print(f"Population: {optim.calculatePopulation(self.G)}")

                applyLSystem()

                if maxRoadSegments is not None and self.G.number_of_edges() >= maxRoadSegments:
                    break

                if maxRoadLength is not None and optim.calculateTotalRoadLength(self.G) > maxRoadLength:
                    break

                if population is not None and optim.calculatePopulation(self.G) > population:
                    break

            if plotType == "Map":
                if plotColour == "Default":
                    graphSettings()
                elif plotColour == "Black":
                    graphSettingsBlack()
                """
                #set the axis limits
                ax.set_xlim([-5,5])
                ax.set_ylim([-5,5])
                """
                
                plt.show()
        
        #return the graph object
        return self.G
    
    def plotCity(self, G, showNodes = False, nodeLabelType = None, edgeLabelType = None):

        fig, ax = plt.subplots()  
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
        edge_colors = ['#ffd747' if G[u][v]['weight'] == 1 else '#e3e3e3' if G[u][v]['weight'] == 0.5 else '#dbdbdb' for u, v in edges]
        edge_widths = [4 if G[u][v]['weight'] == 1 else 3 if G[u][v]['weight'] == 0.5 else 2 for u, v in edges]
        edge_widthsBlack = [x + 1 for x in edge_widths]

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_edges(G, pos, edge_color='black', width=edge_widthsBlack, ax=ax)  # Draw edges with outline
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)  # Draw edges
        if edgeLabelType == "Edge Weight":
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        if edgeLabelType == "Road Number":
            edge_labels = nx.get_edge_attributes(G, 'roadNumber')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)

        if with_labels:
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
        plt.axis('on')  # Turn on the axes
        plt.grid(False)  # Add a grid
        plt.show()


    def plotCityBlack(self, G, showNodes = False, nodeLabelType = None, edgeLabelType = None):

        fig, ax = plt.subplots()  
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
        edge_widths = [1.5 if G[u][v]['weight'] == 1 else 1 if G[u][v]['weight'] == 0.5 else 0.5 for u, v in edges]

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_edges(G, pos, edge_color='black', width=edge_widths, ax=ax)  # Draw edges with outline
        if edgeLabelType == "Edge Weight":
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        if edgeLabelType == "Road Number":
            edge_labels = nx.get_edge_attributes(G, 'roadNumber')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)

        if with_labels:
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)
        plt.axis('on')  # Turn on the axes
        plt.grid(False)  # Add a grid
        plt.show()

    def saveCity(self, G, path):
        with open(path, 'wb') as output:
            pickle.dump(G, output, pickle.HIGHEST_PROTOCOL)

    def loadCity(self, path):
        with open(path, 'rb') as input:
            G = pickle.load(input)
        return G