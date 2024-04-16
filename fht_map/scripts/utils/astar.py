# -*- coding: utf-8 -*-
""" generic A-Star path searching algorithm """

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Union, TypeVar, Generic
from math import inf as infinity
import sortedcontainers  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# introduce generic type
T = TypeVar("T")


################################################################################
class SearchNode(Generic[T]):
    """Representation of a search node"""

    __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "in_openset")

    def __init__(
        self, data: T, gscore: float = infinity, fscore: float = infinity
    ) -> None:
        self.data = data
        self.gscore = gscore
        self.fscore = fscore
        self.closed = False
        self.in_openset = False
        self.came_from: Union[None, SearchNode[T]] = None

    def __lt__(self, b: "SearchNode[T]") -> bool:
        """Natural order is based on the fscore value & is used by heapq operations"""
        return self.fscore < b.fscore


################################################################################
class SearchNodeDict(Dict[T, SearchNode[T]]):
    """A dict that returns a new SearchNode when a key is missing"""

    def __missing__(self, k) -> SearchNode[T]:
        v = SearchNode(k)
        self.__setitem__(k, v)
        return v


################################################################################
SNType = TypeVar("SNType", bound=SearchNode)


class OpenSet(Generic[SNType]):
    def __init__(self) -> None:
        self.sortedlist = sortedcontainers.SortedList(key=lambda x: x.fscore)

    def push(self, item: SNType) -> None:
        item.in_openset = True
        self.sortedlist.add(item)

    def pop(self) -> SNType:
        item = self.sortedlist.pop(0)
        item.in_openset = False
        return item

    def remove(self, item: SNType) -> None:
        self.sortedlist.remove(item)
        item.in_openset = False

    def __len__(self) -> int:
        return len(self.sortedlist)


################################################################################*


class AStar(ABC, Generic[T]):
    __slots__ = ()

    @abstractmethod
    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        """
        Computes the estimated (rough) distance between a node and the goal.
        The second parameter is always the goal.
        This method must be implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def distance_between(self, n1: T, n2: T) -> float:
        """
        Gives the real distance between two adjacent nodes n1 and n2 (i.e n2
        belongs to the list of n1's neighbors).
        n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
        This method must be implemented in a subclass.
        """

    @abstractmethod
    def neighbors(self, node: T) -> Iterable[T]:
        """
        For a given node, returns (or yields) the list of its neighbors.
        This method must be implemented in a subclass.
        """
        raise NotImplementedError

    def is_goal_reached(self, current: T, goal: T) -> bool:
        """
        Returns true when we can consider that 'current' is the goal.
        The default implementation simply compares `current == goal`, but this
        method can be overwritten in a subclass to provide more refined checks.
        """
        return current == goal

    def reconstruct_path(self, last: SearchNode, reversePath=False) -> Iterable[T]:
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from

        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(
        self, start: T, goal: T, reversePath: bool = False
    ) -> Union[Iterable[T], None]:
        if self.is_goal_reached(start, goal):
            return [start]

        openSet: OpenSet[SearchNode[T]] = OpenSet()
        searchNodes: SearchNodeDict[T] = SearchNodeDict()
        startNode = searchNodes[start] = SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal)
        )
        openSet.push(startNode)

        while openSet:
            current = openSet.pop()

            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)

            current.closed = True

            for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                #neighbor is a searchnode object
                if neighbor.closed:
                    continue

                tentative_gscore = current.gscore + self.distance_between(
                    current.data, neighbor.data
                )

                if tentative_gscore >= neighbor.gscore:
                    continue

                neighbor_from_openset = neighbor.in_openset

                if neighbor_from_openset:
                    # we have to remove the item from the heap, as its score has changed
                    openSet.remove(neighbor)

                # update the node
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(
                    neighbor.data, goal
                )

                openSet.push(neighbor)

        return None

    def astar_multigoal(
        self, start: T, goal: T, reversePath: bool = False
    ) -> Union[Iterable[T], None]:

        openSet: OpenSet[SearchNode[T]] = OpenSet()
        searchNodes: SearchNodeDict[T] = SearchNodeDict()
        startNode = searchNodes[start] = SearchNode(
            start, gscore=0.0, fscore=self.heuristic_cost_estimate(start, goal[0])
        )
        openSet.push(startNode)

        result_path = []
        path_length = []
        for now_goal in goal:
            if now_goal in searchNodes.keys():
                result_path.append(list(self.reconstruct_path(searchNodes[now_goal], reversePath)))
                path_length.append(searchNodes[now_goal].gscore)
                continue

            openSet = OpenSet()
            for now_key in searchNodes.keys():
                now_node = searchNodes[now_key]
                now_node.fscore = now_node.gscore + self.heuristic_cost_estimate(now_node.data, now_goal)
                openSet.push(now_node)
            reachable_flag = False
            while openSet:
                current = openSet.pop()
                if self.is_goal_reached(current.data, now_goal):
                    result_path.append(list(self.reconstruct_path(current, reversePath)))
                    path_length.append(current.gscore)
                    reachable_flag = True
                    break

                current.closed = True
                for neighbor in map(lambda n: searchNodes[n], self.neighbors(current.data)):
                    #neighbor is a searchnode object
                    if neighbor.closed:
                        continue
                    tentative_gscore = current.gscore + self.distance_between(
                        current.data, neighbor.data
                    )

                    if tentative_gscore >= neighbor.gscore:
                        #this is a longer road
                        continue
                    neighbor_from_openset = neighbor.in_openset

                    if neighbor_from_openset:
                        # we have to remove the item from the heap, as its score has changed
                        openSet.remove(neighbor)

                    # update the node
                    neighbor.came_from = current
                    neighbor.gscore = tentative_gscore
                    neighbor.fscore = tentative_gscore + self.heuristic_cost_estimate(neighbor.data, now_goal)

                    openSet.push(neighbor)
            if not reachable_flag:
                result_path.append(None)
                path_length.append(1e10)
        return result_path, path_length

################################################################################*

class grid_path(AStar):
    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""
    #using (y,x) as input
    def __init__(self, grid_map,distance_map,start_point,end_point):
        self.grid_map = grid_map
        self.distance_map = distance_map
        self.width = grid_map.shape[0]
        self.height = grid_map.shape[1]
        self.start_point = start_point
        self.end_point = end_point

        self.foundPath = None
        self.path_length = None

    def get_path(self):
        self.foundPath, self.path_length = self.astar_multigoal(self.start_point,self.end_point )

    def vis_path(self):
        plt.imshow(self.grid_map, cmap='gray')
        plt.scatter(self.start_point[1],self.start_point[0])
        
        for now_path in self.foundPath:
            now_path = np.array(now_path)
            # 打印矩阵
            plt.scatter(now_path[:,1],now_path[:,0],1)
            
        for now_end in self.end_point:
            plt.scatter(now_end[1],now_end[0])
        plt.show()
    
    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        if self.distance_map[n1[0],n1[1]] < 10:
            return ((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)**0.5
        else:
            return ((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)**0.5 - 40

    def distance_between(self, n1, n2):
        if n1[0] == n2[0] or n1[1]  == n2[1]:
            return 1
        else:
            return 1.414

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        return[(nx, ny) for nx, ny in[(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y), (x + 1, y),(x-1, y - 1), (x - 1, y + 1), (x + 1, y -1), (x + 1, y +1)]
        if 0 <= nx < self.width and 0 <= ny < self.height and self.grid_map[nx, ny] ==0  ]

################################################################################*
class topo_map_path(AStar):
    def __init__(self, nodes, start_point, end_point):
        self.nodes = nodes
        self.start_point = start_point
        self.end_point = end_point
        self.foundPath = None
        self.path_length = None

    def get_path(self):
        self.foundPath, self.path_length = self.astar_multigoal(self.start_point,self.end_point )

    def neighbors(self, n):
        for n1, d in self.nodes[n]:
            yield n1

    def distance_between(self, n1, n2):
        for n, d in self.nodes[n1]:
            if n == n2:
                return d
            
    def heuristic_cost_estimate(self, current, goal):
        return 1
    
    def is_goal_reached(self, current, goal):
        return current == goal