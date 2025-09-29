#!/usr/bin/env python3
"""
Simple TSP Problem Generator and Solver
No external dependencies required - uses only Python standard library
"""

import random
import math
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ProblemType(Enum):
    METRIC_TSP = "Metric TSP"
    GENERAL_TSP = "General TSP"
    MULTI_TSP = "Multi-Salesman TSP"

@dataclass
class TSPProblem:
    """Simple TSP Problem representation"""
    problem_type: ProblemType
    n_nodes: int
    nodes: Optional[List[List[float]]] = None
    edges: Optional[List[List[float]]] = None
    n_salesmen: int = 1
    depots: Optional[List[int]] = None
    
    def __post_init__(self):
        """Initialize problem after creation"""
        if self.problem_type == ProblemType.METRIC_TSP and not self.nodes:
            self.nodes = self._generate_random_coordinates()
            self.edges = self._calculate_distance_matrix()
        elif self.problem_type == ProblemType.GENERAL_TSP and not self.edges:
            self.edges = self._generate_random_edges()
        elif self.problem_type == ProblemType.MULTI_TSP:
            if not self.depots:
                self.depots = [0] * self.n_salesmen
            if not self.nodes:
                self.nodes = self._generate_random_coordinates()
                self.edges = self._calculate_distance_matrix()
    
    def _generate_random_coordinates(self, grid_size: int = 1000) -> List[List[float]]:
        """Generate random city coordinates"""
        coordinates = []
        for _ in range(self.n_nodes):
            x = random.uniform(0, grid_size)
            y = random.uniform(0, grid_size)
            coordinates.append([x, y])
        return coordinates
    
    def _calculate_distance_matrix(self) -> List[List[float]]:
        """Calculate Euclidean distance matrix"""
        if not self.nodes:
            return []
        
        n = len(self.nodes)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.nodes[i][0] - self.nodes[j][0]
                    dy = self.nodes[i][1] - self.nodes[j][1]
                    distance_matrix[i][j] = math.sqrt(dx*dx + dy*dy)
        
        return distance_matrix
    
    def _generate_random_edges(self, min_weight: int = 1, max_weight: int = 100) -> List[List[float]]:
        """Generate random edge weights"""
        edges = [[0.0 for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    edges[i][j] = random.uniform(min_weight, max_weight)
        
        return edges
    
    def get_info(self) -> dict:
        """Get problem information"""
        return {
            "Problem Type": self.problem_type.value,
            "Number of Nodes": self.n_nodes,
            "Number of Salesmen": self.n_salesmen,
            "Has Coordinates": self.nodes is not None,
            "Has Edge Matrix": self.edges is not None,
            "Depots": self.depots
        }

class TSPSolver:
    """Simple TSP Solver using nearest neighbor heuristic"""
    
    def solve(self, problem: TSPProblem) -> Union[List[int], List[List[int]]]:
        """Solve TSP problem"""
        if problem.problem_type == ProblemType.MULTI_TSP:
            return self._solve_multi_tsp(problem)
        else:
            return self._solve_single_tsp(problem)
    
    def _solve_single_tsp(self, problem: TSPProblem) -> List[int]:
        """Solve single TSP using nearest neighbor"""
        if not problem.edges:
            return []
        
        n = len(problem.edges)
        visited = [False] * n
        route = []
        
        # Start from node 0
        current_node = 0
        route.append(current_node)
        visited[current_node] = True
        
        # Visit remaining nodes
        for _ in range(n - 1):
            nearest_distance = float('inf')
            nearest_node = None
            
            for j in range(n):
                if not visited[j] and problem.edges[current_node][j] < nearest_distance:
                    nearest_distance = problem.edges[current_node][j]
                    nearest_node = j
            
            if nearest_node is not None:
                route.append(nearest_node)
                visited[nearest_node] = True
                current_node = nearest_node
        
        # Return to start
        route.append(route[0])
        return route
    
    def _solve_multi_tsp(self, problem: TSPProblem) -> List[List[int]]:
        """Solve multi-salesman TSP using greedy assignment"""
        if not problem.edges or not problem.depots:
            return []
        
        n = len(problem.edges)
        n_salesmen = problem.n_salesmen
        depots = problem.depots
        
        # Initialize routes
        routes = [[] for _ in range(n_salesmen)]
        visited = [False] * n
        
        # Start each salesman at their depot
        for i, depot in enumerate(depots):
            routes[i].append(depot)
            visited[depot] = True
        
        # Assign remaining nodes greedily
        remaining_nodes = [i for i in range(n) if not visited[i]]
        
        while remaining_nodes:
            best_assignment = None
            best_cost = float('inf')
            
            for node in remaining_nodes:
                for salesman_idx in range(n_salesmen):
                    # Calculate cost of adding this node to this salesman's route
                    current_route = routes[salesman_idx]
                    if len(current_route) == 0:
                        cost = 0
                    else:
                        cost = problem.edges[current_route[-1]][node]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = (node, salesman_idx)
            
            if best_assignment:
                node, salesman_idx = best_assignment
                routes[salesman_idx].append(node)
                visited[node] = True
                remaining_nodes.remove(node)
            else:
                # Assign to first available salesman
                if remaining_nodes:
                    node = remaining_nodes.pop(0)
                    routes[0].append(node)
                    visited[node] = True
        
        # Complete routes by returning to depots
        for i, route in enumerate(routes):
            if len(route) > 0:
                route.append(route[0])
        
        return routes

def calculate_solution_cost(problem: TSPProblem, solution: Union[List[int], List[List[int]]]) -> float:
    """Calculate the cost of a solution"""
    if not problem.edges:
        return 0.0
    
    if isinstance(solution, list) and len(solution) > 0 and isinstance(solution[0], list):
        # Multi-salesman solution
        total_cost = 0.0
        for route in solution:
            if len(route) > 1:
                for i in range(len(route) - 1):
                    total_cost += problem.edges[route[i]][route[i + 1]]
        return total_cost
    else:
        # Single route solution
        if len(solution) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(solution) - 1):
            total_cost += problem.edges[solution[i]][solution[i + 1]]
        return total_cost

def create_problem_generator():
    """Create a problem generator"""
    class ProblemGenerator:
        def generate_metric_tsp(self, n_nodes: int) -> TSPProblem:
            """Generate metric TSP problem"""
            return TSPProblem(
                problem_type=ProblemType.METRIC_TSP,
                n_nodes=n_nodes
            )
        
        def generate_general_tsp(self, n_nodes: int) -> TSPProblem:
            """Generate general TSP problem"""
            return TSPProblem(
                problem_type=ProblemType.GENERAL_TSP,
                n_nodes=n_nodes
            )
        
        def generate_multi_tsp(self, n_nodes: int, n_salesmen: int) -> TSPProblem:
            """Generate multi-salesman TSP problem"""
            return TSPProblem(
                problem_type=ProblemType.MULTI_TSP,
                n_nodes=n_nodes,
                n_salesmen=n_salesmen
            )
    
    return ProblemGenerator()

def main():
    """Main function to demonstrate the TSP system"""
    print("=== Simple TSP Problem Generator and Solver ===\n")
    
    # Create generator and solver
    generator = create_problem_generator()
    solver = TSPSolver()
    
    # Test different problem types
    problems = [
        ("Metric TSP (10 nodes)", generator.generate_metric_tsp(10)),
        ("General TSP (8 nodes)", generator.generate_general_tsp(8)),
        ("Multi-Salesman TSP (12 nodes, 2 salesmen)", generator.generate_multi_tsp(12, 2)),
        ("Multi-Salesman TSP (15 nodes, 3 salesmen)", generator.generate_multi_tsp(15, 3))
    ]
    
    for name, problem in problems:
        print(f"=== {name} ===")
        
        # Show problem info
        info = problem.get_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Solve the problem
        print("Solving...")
        solution = solver.solve(problem)
        cost = calculate_solution_cost(problem, solution)
        
        if isinstance(solution, list) and len(solution) > 0 and isinstance(solution[0], list):
            # Multi-salesman solution
            print(f"Solution: {len(solution)} routes")
            for i, route in enumerate(solution):
                print(f"  Route {i+1}: {route}")
        else:
            # Single route solution
            print(f"Solution: {solution}")
        
        print(f"Total Cost: {cost:.2f}")
        print()

if __name__ == "__main__":
    main()
