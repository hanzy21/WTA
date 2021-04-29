from __future__ import division
import random
import math
import time 
import numpy as np

class RRTPlanner():
    
    def __init__(self, start, goal, obstacle):
        self.start = start
        self.goal = goal
        self.obstacle = obstacle
        self.V = set()
        self.E = set()
        self.steer_distance = 50 #每一步延长的距离
        self.enough_dist = 50 #视作到达终点的范围
        self.child_to_parent_dict = dict()
        
    def RRT(self):    
        path=[]
        N = 10000 #循环次数
        if self.start == self.goal:
            path = [self.start, self.goal]
            self.V.union([self.start, self.goal])
            self.E.union([(self.start, self.goal)])           
            return self.length(path),path
        elif self.isEdgeCollisionFree(self.start, self.goal):
            path = [self.start, self.goal]
            self.V.union([self.start, self.goal])
            self.E.union([(self.start, self.goal)])
            return self.length(path),path
        else:
            self.V.add(self.start)
            for i in range(N):
                random_point = self.get_collision_free_random_point()
                nearest_point = self.find_nearest_point(random_point)
                new_point = self.steer(nearest_point, random_point)
                if self.isEdgeCollisionFree(nearest_point, new_point):
                    self.V.add(new_point)
                    self.E.add((nearest_point, new_point))
                    self.setParent(nearest_point, new_point)
                    # If new point of the tree is at the goal region, we can find a path in the tree from start node to goal.
                    if self.isAtGoalRegion(new_point): # If not running for full iterations, terminate as soon as a path is found.
                        path = self.find_path(self.start, new_point)          
                        break
            # If no path is found, then path would be an empty list.
            if path == []:
                print("a")
                return 10*self.length([self.start,self.goal]),None #如果规定次数内没规划出来，返回十倍的欧氏距离
            else:
                uniPruningPath = self.uniPruning(path)
                return self.length(uniPruningPath),uniPruningPath
                #return self.length(path), path
                
    def isEdgeCollisionFree(self,point1, point2):
        points = self.es_points_along_line(point1, point2)
        for i in points:
            if not self.isPointCollisionFree(i):
                return False
        return True

    
    def isPointCollisionFree(self,point):
        i = int(point[1])
        j = int(point[0])
        if self.obstacle[i][j] == 1 or i > self.obstacle.shape[0] or j > self.obstacle.shape[1]:
            return False
        return True
    
    def get_collision_free_random_point(self):
        # Run until a valid point is found
        while True:
            point = self.get_random_point()
            if self.isPointCollisionFree(point):
                return point
            
    def get_random_point(self):
        x = random.random() * self.obstacle.shape[0]
        y = random.random() * self.obstacle.shape[1]
        return (x, y)
    
    def steer(self, from_point, to_point):
        if self.euclidian_dist(from_point, to_point) < self.steer_distance:
            return to_point
        else:
            from_x, from_y = from_point
            to_x, to_y = to_point
            theta = math.atan2(to_y - from_y, to_x- from_x)
            new_point = (int(from_x + self.steer_distance * math.cos(theta)), int(from_y + self.steer_distance * math.sin(theta)))
            return new_point
        
    def find_path(self, start_point, end_point):
        # Returns a path by backtracking through the tree formed by one of the RRT algorithms starting at the end_point until reaching start_node.
        path = [end_point]
        tree_size, path_size, path_length = len(self.V), 1, 0
        current_node = end_point
        previous_node = None
        target_node = start_point
        while current_node != target_node:
            parent = self.getParent(current_node)
            path.append(parent)
            previous_node = current_node
            current_node = parent
            path_length += self.euclidian_dist(current_node, previous_node)
            path_size += 1
        path.reverse()
        return path #, tree_size, path_size, path_length        
        
    def find_nearest_point(self, random_point):
        closest_point = None
        min_dist = float('inf')
        for vertex in self.V:
            euc_dist = self.euclidian_dist(random_point, vertex)
            if euc_dist < min_dist:
                min_dist = euc_dist
                closest_point = vertex
        return closest_point

    def euclidian_dist(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def getParent(self, vertex):
        return self.child_to_parent_dict[vertex]

    def setParent(self, parent, child):
        self.child_to_parent_dict[child] = parent
        
    def isAtGoalRegion(self,point):
        if self.euclidian_dist(point,self.goal) <= self.enough_dist:
            return True
        return False

    def uniPruning(self, path):     #Pruning function
        unidirectionalPath=[path[0]]
        pointTem=path[0]
        num = 0
        for i in range(3,len(path)):
            num += 1
            if not self.isEdgeCollisionFree(pointTem,path[i]):
                pointTem=path[i-1]
                unidirectionalPath.append(pointTem)
                num = 0
            #if num == 20:
             #   pointTem=path[i-1]
              #  unidirectionalPath.append(pointTem)     
               # num = 0
        unidirectionalPath.append(path[-1])
        return unidirectionalPath
    
    def length(self,path):
        path_length = 0
        for i in range(1,len(path)):
            path_length += self.euclidian_dist(path[i], path[i-1])
        return path_length
    
    def es_points_along_line(self,start, end):
        d = self.euclidian_dist(start, end)
        start, end = np.array(start), np.array(end)
        r = 10
        n_points = int(d/r)        
        pointlist = [start, end]
        if n_points > 1:
            step = d / (n_points - 1)
            for i in range(n_points):
                v = end - start
                u = v / (np.sqrt(np.sum(v ** 2)))
                steered_point = start + u * i * step
                pointlist.append(steered_point)
        return pointlist
    
    
    
    
    
    