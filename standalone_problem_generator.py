#!/usr/bin/env python3
"""
Standalone TSP Problem Generator
Creates TSP problems without Bittensor dependencies
"""

import numpy as np
import random
import math
from typing import List, Union, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from enum import Enum

class ProblemType(str, Enum):
    METRIC_TSP = "Metric TSP"
    GENERAL_TSP = "General TSP"
    METRIC_MTSP = "Metric mTSP"
    GENERAL_MTSP = "General mTSP"
    METRIC_CMTSP = "Metric cmTSP"
    GENERAL_CMTSP = "General cmTSP"
    METRIC_CMTSPTW = "Metric cmTSPTW"
    GENERAL_CMTSPTW = "General cmTSPTW"
    PORTFOLIO = "PortfolioReallocation"

class StandaloneTSPProblem(BaseModel):
    """Standalone TSP Problem without Bittensor dependencies"""
    problem_type: ProblemType = Field(ProblemType.METRIC_TSP, description="Problem Type")
    objective_function: str = Field('min', description="Objective Function")
    visit_all: bool = Field(True, description="Visit All Nodes")
    to_origin: bool = Field(True, description="Return to Origin")
    n_nodes: int = Field(10, ge=2, description="Number of Nodes")
    nodes: Optional[List[List[Union[int, float]]]] = Field(None, description="Node Coordinates")
    edges: Optional[List[List[Union[int, float]]]] = Field(None, description="Edge Weights")
    directed: bool = Field(False, description="Directed Graph")
    simple: bool = Field(True, description="Simple Graph")
    weighted: bool = Field(False, description="Weighted Graph")
    repeating: bool = Field(False, description="Allow Repeating Nodes")
    
    # Multi-salesman specific fields
    n_salesmen: Optional[int] = Field(None, ge=2, le=10, description="Number of Salesmen")
    single_depot: bool = Field(True, description="Single depot formulation")
    depots: Optional[List[int]] = Field(None, description="Depot locations")
    
    # Constrained TSP fields
    demand: Optional[List[int]] = Field(None, description="Node demands")
    constraint: Optional[List[int]] = Field(None, description="Vehicle capacity constraints")
    
    # Time window fields
    time_windows: Optional[List[Tuple[Union[int, float], Union[int, float]]]] = Field(None, description="Time windows")
    
    # Portfolio fields
    n_portfolio: Optional[int] = Field(None, description="Number of portfolios")
    initial_portfolios: Optional[List[List[int]]] = Field(None, description="Initial portfolio allocations")
    constraint_values: Optional[List[Union[float, int]]] = Field(None, description="Constraint values")
    constraint_types: Optional[List[str]] = Field(None, description="Constraint types")
    pools: Optional[List[List[int]]] = Field(None, description="Pool states")

    @model_validator(mode='after')
    def initialize_nodes_and_edges(self):
        """Generate nodes and edges if not provided"""
        if not self.directed and not self.nodes and not self.edges:
            # Generate random coordinates for metric TSP
            self.nodes = self.generate_random_coordinates(self.n_nodes)
            self.edges = self.get_distance_matrix(self.nodes)
        elif self.directed and not self.edges:
            # Generate random edge weights for general TSP
            self.problem_type = ProblemType.GENERAL_TSP
            self.edges = self.generate_edges(self.n_nodes)
        elif not self.directed and self.nodes and not self.edges:
            # Calculate distance matrix from coordinates
            self.edges = self.get_distance_matrix(self.nodes)
        return self

    @model_validator(mode='after')
    def validate_multi_salesman(self):
        """Validate multi-salesman problem setup"""
        if self.problem_type in [ProblemType.METRIC_MTSP, ProblemType.GENERAL_MTSP, 
                                ProblemType.METRIC_CMTSP, ProblemType.GENERAL_CMTSP,
                                ProblemType.METRIC_CMTSPTW, ProblemType.GENERAL_CMTSPTW]:
            if self.n_salesmen is None:
                self.n_salesmen = 2
            if self.depots is None:
                if self.single_depot:
                    self.depots = [0] * self.n_salesmen
                else:
                    self.depots = sorted(random.sample(range(self.n_nodes), self.n_salesmen))
        return self

    def generate_random_coordinates(self, n_cities: int, grid_size: int = 1000) -> List[List[int]]:
        """Generate random city coordinates"""
        x = np.arange(grid_size)
        y = np.arange(grid_size)
        xv, yv = np.meshgrid(x, y)
        
        coordinates = np.column_stack((xv.ravel(), yv.ravel()))
        sampled_indices = np.random.choice(coordinates.shape[0], n_cities, replace=False)
        sampled_coordinates = coordinates[sampled_indices]
        np.random.shuffle(sampled_coordinates)
        
        return [[int(coord) for coord in pair] for pair in sampled_coordinates.tolist()]

    def get_distance_matrix(self, coordinates: List[List[int]]) -> List[List[float]]:
        """Calculate Euclidean distance matrix"""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    distance_matrix[i, j] = np.sqrt(
                        (coordinates[i][0] - coordinates[j][0])**2 + 
                        (coordinates[i][1] - coordinates[j][1])**2
                    )
        return distance_matrix.tolist()

    def generate_edges(self, num_cities: int, min_weight: int = 1, max_weight: int = 100) -> List[List[int]]:
        """Generate random edge weights for general TSP"""
        edges = np.zeros((num_cities, num_cities), dtype=int)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    weight = np.random.randint(min_weight, max_weight)
                    edges[i][j] = weight
        return edges.tolist()

    def get_info(self, verbosity: int = 1) -> dict:
        """Get problem information"""
        info = {}
        if verbosity >= 1:
            info["Problem Type"] = self.problem_type
            info["Number of Nodes"] = self.n_nodes
        if verbosity >= 2:
            info["Objective Function"] = self.objective_function
            info["Visit All Nodes"] = self.visit_all
            info["Return to Origin"] = self.to_origin
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
            if self.n_salesmen:
                info["Number of Salesmen"] = self.n_salesmen
                info["Single Depot"] = self.single_depot
                info["Depots"] = self.depots
        if verbosity >= 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

class StandaloneProblemGenerator:
    """Generator for creating various TSP problems"""
    
    def __init__(self, datasets: Optional[dict] = None):
        self.datasets = datasets or {}
        self.load_default_datasets()
    
    def load_default_datasets(self):
        """Load default TSP datasets"""
        try:
            # Try to load Asia_MSB dataset
            with np.load('dataset/Asia_MSB.npz') as f:
                node_coords_np = f['data']
                self.datasets["Asia_MSB"] = np.array(node_coords_np)
        except FileNotFoundError:
            print("Warning: Asia_MSB dataset not found. Using synthetic data.")
            # Create synthetic dataset
            synthetic_data = np.random.rand(1000, 3) * 1000
            self.datasets["Asia_MSB"] = synthetic_data

    def generate_metric_tsp(self, n_nodes: int = 100, dataset_ref: str = "Asia_MSB") -> StandaloneTSPProblem:
        """Generate a metric TSP problem"""
        if dataset_ref in self.datasets:
            # Select random nodes from dataset
            dataset_size = len(self.datasets[dataset_ref])
            selected_indices = random.sample(range(dataset_size), min(n_nodes, dataset_size))
            nodes = [[float(coord) for coord in self.datasets[dataset_ref][idx][1:]] 
                    for idx in selected_indices]
        else:
            nodes = None  # Will be generated automatically
        
        return StandaloneTSPProblem(
            problem_type=ProblemType.METRIC_TSP,
            n_nodes=n_nodes,
            nodes=nodes
        )

    def generate_general_tsp(self, n_nodes: int = 100) -> StandaloneTSPProblem:
        """Generate a general TSP problem with random edge weights"""
        return StandaloneTSPProblem(
            problem_type=ProblemType.GENERAL_TSP,
            n_nodes=n_nodes,
            directed=True
        )

    def generate_multi_tsp(self, n_nodes: int = 50, n_salesmen: int = 2, 
                          single_depot: bool = True, dataset_ref: str = "Asia_MSB") -> StandaloneTSPProblem:
        """Generate a multi-salesman TSP problem"""
        if dataset_ref in self.datasets:
            dataset_size = len(self.datasets[dataset_ref])
            selected_indices = random.sample(range(dataset_size), min(n_nodes, dataset_size))
            nodes = [[float(coord) for coord in self.datasets[dataset_ref][idx][1:]] 
                    for idx in selected_indices]
        else:
            nodes = None
        
        depots = [0] * n_salesmen if single_depot else sorted(random.sample(range(n_nodes), n_salesmen))
        
        return StandaloneTSPProblem(
            problem_type=ProblemType.METRIC_MTSP,
            n_nodes=n_nodes,
            n_salesmen=n_salesmen,
            single_depot=single_depot,
            depots=depots,
            nodes=nodes
        )

    def generate_constrained_multi_tsp(self, n_nodes: int = 50, n_salesmen: int = 2,
                                     dataset_ref: str = "Asia_MSB") -> StandaloneTSPProblem:
        """Generate a constrained multi-salesman TSP problem"""
        if dataset_ref in self.datasets:
            dataset_size = len(self.datasets[dataset_ref])
            selected_indices = random.sample(range(dataset_size), min(n_nodes, dataset_size))
            nodes = [[float(coord) for coord in self.datasets[dataset_ref][idx][1:]] 
                    for idx in selected_indices]
        else:
            nodes = None
        
        depots = sorted(random.sample(range(n_nodes), n_salesmen))
        demand = [random.randint(1, 9) for _ in range(n_nodes)]
        for depot in depots:
            demand[depot] = 0
        
        # Generate capacity constraints
        total_demand = sum(demand)
        constraint = []
        for i in range(n_salesmen - 1):
            constraint.append(random.randint(total_demand // n_salesmen, total_demand))
        constraint.append(total_demand - sum(constraint))
        
        return StandaloneTSPProblem(
            problem_type=ProblemType.METRIC_CMTSP,
            n_nodes=n_nodes,
            n_salesmen=n_salesmen,
            single_depot=False,
            depots=depots,
            demand=demand,
            constraint=constraint,
            nodes=nodes
        )

    def generate_time_window_tsp(self, n_nodes: int = 50, n_salesmen: int = 2,
                               dataset_ref: str = "Asia_MSB") -> StandaloneTSPProblem:
        """Generate a time-window constrained TSP problem"""
        # First generate a constrained mTSP
        problem = self.generate_constrained_multi_tsp(n_nodes, n_salesmen, dataset_ref)
        
        # Add time windows
        time_windows = []
        for i in range(n_nodes):
            start_time = random.uniform(0, 50)
            end_time = start_time + random.uniform(10, 30)
            time_windows.append((start_time, end_time))
        
        problem.problem_type = ProblemType.METRIC_CMTSPTW
        problem.time_windows = time_windows
        
        return problem

    def generate_portfolio_problem(self, n_portfolio: int = 10, n_subnets: int = 10) -> StandaloneTSPProblem:
        """Generate a portfolio reallocation problem"""
        # Generate random initial portfolios
        initial_portfolios = []
        for _ in range(n_portfolio):
            portfolio = [random.randint(0, 100) for _ in range(n_subnets)]
            initial_portfolios.append(portfolio)
        
        # Generate constraint types and values
        constraint_types = [random.choice(["eq", "ge", "le"]) for _ in range(n_subnets)]
        constraint_values = []
        for ctype in constraint_types:
            if ctype == "eq":
                constraint_values.append(random.uniform(0.5, 3.0))
            elif ctype == "ge":
                constraint_values.append(random.uniform(0.0, 5.0))
            else:  # le
                constraint_values.append(random.uniform(10.0, 100.0))
        
        # Generate pool states
        pools = [[random.uniform(1.0, 10.0), random.uniform(1.0, 10.0)] for _ in range(n_subnets)]
        
        return StandaloneTSPProblem(
            problem_type=ProblemType.PORTFOLIO,
            n_portfolio=n_portfolio,
            initial_portfolios=initial_portfolios,
            constraint_values=constraint_values,
            constraint_types=constraint_types,
            pools=pools
        )

def main():
    """Example usage of the standalone problem generator"""
    print("=== Standalone TSP Problem Generator ===\n")
    
    generator = StandaloneProblemGenerator()
    
    # Generate different types of problems
    problems = [
        ("Metric TSP", generator.generate_metric_tsp(20)),
        ("General TSP", generator.generate_general_tsp(20)),
        ("Multi-Salesman TSP", generator.generate_multi_tsp(15, 3)),
        ("Constrained Multi-Salesman TSP", generator.generate_constrained_multi_tsp(15, 3)),
        ("Time-Window TSP", generator.generate_time_window_tsp(15, 3)),
        ("Portfolio Problem", generator.generate_portfolio_problem(5, 5))
    ]
    
    for name, problem in problems:
        print(f"=== {name} ===")
        info = problem.get_info(verbosity=2)
        for key, value in info.items():
            print(f"{key}: {value}")
        print(f"Edges shape: {len(problem.edges) if problem.edges else 'None'}x{len(problem.edges[0]) if problem.edges and problem.edges[0] else 'None'}")
        print()

if __name__ == "__main__":
    main()
