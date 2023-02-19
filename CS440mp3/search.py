# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq

from state import MazeState
from maze import Maze

# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze
def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)

def astar(maze, ispart1=False):
    """
    This function returns an optimal path in a list, which contains the start and objective.

    @param maze: Maze instance from maze.py
    @param ispart1:pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    @return: a path in the form of a list of MazeState objects. If there is no path, return None.
    """
    # Your code here
    # Construct a state
    start_state = MazeState(maze.getStart().state, maze.getObjectives(), 0, maze, mst_cache={}, use_heuristic=True)
    visited_states = {start_state: (None, 0)}

    frontier = []
    heapq.heappush(frontier, start_state)

    while(frontier):#while it is not empty
        curr = heapq.heappop(frontier)
        if (curr.is_goal()): #check if the search is finished
            shortest_path = backtrack(visited_states, curr)
            
            return shortest_path
        
        neighbors = curr.get_neighbors(ispart1) #find its neighbors
        #push all its neighbors to the hq
        for i in neighbors:
            nei_dist = i.dist_from_start # dist to the neighbor
            if (i not in visited_states): # if the neighbor not seen before
                visited_states[i] = (curr, nei_dist) # add the state's parent and dist_from_start to hq
                heapq.heappush(frontier, i) #push to hq
            elif (nei_dist < visited_states[i][1]): # if it finds a shorter dist, renew it and push it to the hq
                visited_states[i] = (curr, nei_dist) # add the state's parent and dist_from_start to hq
                heapq.heappush(frontier, i) #push to hq
            
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return None

# This is the same as backtrack from MP2
def backtrack(visited_states, current_state):
    path = []
    # Your code here ---------------
    #Initialization
    curr = current_state
    path.append(curr) #add the goal state to list

    while(visited_states[curr][0] is not None): #keep backtrack if it doesn't reach the start state
        path.insert(0, visited_states[curr][0]) #add parent state to the list
        curr = visited_states[curr][0] #renew curr state
    return path
        