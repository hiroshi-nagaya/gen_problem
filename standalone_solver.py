#!/usr/bin/env python3
"""
Standalone TSP Solver
Solves TSP problems without Bittensor dependencies
"""

import numpy as np
import asyncio
import random
import math
from typing import List, Union, Optional, Tuple
from abc import ABC, abstractmethod
from standalone_problem_generator import StandaloneTSPProblem, ProblemType


class BaseStandaloneSolver(ABC):
    """Base class for standalone TSP solvers"""

    def __init__(self, problem_types: List[ProblemType]):
        self.problem_types = problem_types
        self.future_tracker = {}

    @abstractmethod
    async def solve(
        self, problem: StandaloneTSPProblem
    ) -> Union[List[int], List[List[int]], bool]:
        """Solve the TSP problem"""
        pass

    def is_valid_problem(self, problem: StandaloneTSPProblem) -> bool:
        """Check if solver can handle this problem type"""
        return problem.problem_type in self.problem_types


class NearestNeighborSolver(BaseStandaloneSolver):
    """Nearest Neighbor solver for basic TSP"""

    def __init__(self):
        super().__init__([ProblemType.METRIC_TSP, ProblemType.GENERAL_TSP])

    async def solve(self, problem: StandaloneTSPProblem) -> List[int]:
        """Solve using nearest neighbor heuristic"""
        if not self.is_valid_problem(problem):
            raise ValueError(
                f"Solver cannot handle problem type: {problem.problem_type}")

        distance_matrix = np.array(problem.edges)
        n = len(distance_matrix)
        visited = [False] * n
        route = []

        # Start from node 0
        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            # Find nearest unvisited neighbor
            nearest_distance = np.inf
            nearest_node = None

            for j in range(n):
                if not visited[j] and distance_matrix[current_node][
                        j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            if nearest_node is not None:
                route.append(nearest_node)
                visited[nearest_node] = True
                current_node = nearest_node

        # Return to start if required
        if problem.to_origin:
            route.append(route[0])

        return route


class BeamSearchSolver(BaseStandaloneSolver):
    """Beam Search solver for TSP"""

    def __init__(self, beam_width: int = 3):
        super().__init__([ProblemType.METRIC_TSP, ProblemType.GENERAL_TSP])
        self.beam_width = beam_width

    async def solve(self, problem: StandaloneTSPProblem) -> List[int]:
        """Solve using beam search"""
        if not self.is_valid_problem(problem):
            raise ValueError(
                f"Solver cannot handle problem type: {problem.problem_type}")

        distance_matrix = np.array(problem.edges)
        n = len(distance_matrix)

        # Initialize beam with starting point
        beam = [(0, [0], 0)]  # (current_node, path, total_distance)

        for _ in range(n - 1):
            candidates = []

            # Expand each path in the beam
            for current_node, path, current_distance in beam:
                for next_node in range(n):
                    if next_node not in path:
                        new_path = path + [next_node]
                        new_distance = current_distance + distance_matrix[
                            current_node][next_node]
                        candidates.append((next_node, new_path, new_distance))

            # Sort by distance and keep top beam_width
            candidates.sort(key=lambda x: x[2])
            beam = candidates[:min(self.beam_width, len(candidates))]

        # Complete the tour
        final_candidates = []
        for current_node, path, current_distance in beam:
            if problem.to_origin:
                final_distance = current_distance + distance_matrix[
                    current_node][0]
                final_candidates.append((path + [0], final_distance))
            else:
                final_candidates.append((path, current_distance))

        # Return best solution
        best_path, _ = min(final_candidates, key=lambda x: x[1])
        return best_path


class MultiSalesmanSolver(BaseStandaloneSolver):
    """Solver for multi-salesman TSP problems"""

    def __init__(self):
        super().__init__([
            ProblemType.METRIC_MTSP, ProblemType.GENERAL_MTSP,
            ProblemType.METRIC_CMTSP, ProblemType.GENERAL_CMTSP,
            ProblemType.METRIC_CMTSPTW, ProblemType.GENERAL_CMTSPTW
        ])

    async def solve(self, problem: StandaloneTSPProblem) -> List[List[int]]:
        """Solve multi-salesman TSP using greedy assignment"""
        if not self.is_valid_problem(problem):
            raise ValueError(
                f"Solver cannot handle problem type: {problem.problem_type}")

        distance_matrix = np.array(problem.edges)
        n = len(distance_matrix)
        n_salesmen = problem.n_salesmen
        depots = problem.depots

        # Initialize routes for each salesman
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
            best_cost = np.inf

            for node in remaining_nodes:
                for salesman_idx in range(n_salesmen):
                    if self._is_valid_assignment(problem, routes[salesman_idx],
                                                 node):
                        # Calculate cost of adding this node to this salesman's route
                        current_route = routes[salesman_idx]
                        if len(current_route) == 1:
                            cost = distance_matrix[current_route[-1]][node]
                        else:
                            # Insert at best position
                            cost = self._find_best_insertion_cost(
                                distance_matrix, current_route, node)

                        if cost < best_cost:
                            best_cost = cost
                            best_assignment = (node, salesman_idx)

            if best_assignment:
                node, salesman_idx = best_assignment
                routes[salesman_idx].append(node)
                visited[node] = True
                remaining_nodes.remove(node)
            else:
                # If no valid assignment, assign to first available salesman
                if remaining_nodes:
                    node = remaining_nodes.pop(0)
                    routes[0].append(node)
                    visited[node] = True

        # Complete routes by returning to depots
        for i, route in enumerate(routes):
            if problem.to_origin and len(route) > 1:
                route.append(route[0])

        return routes

    def _is_valid_assignment(self, problem: StandaloneTSPProblem,
                             route: List[int], node: int) -> bool:
        """Check if assigning a node to a route is valid"""
        if problem.problem_type in [
                ProblemType.METRIC_CMTSP, ProblemType.GENERAL_CMTSP,
                ProblemType.METRIC_CMTSPTW, ProblemType.GENERAL_CMTSPTW
        ]:
            # Check capacity constraints
            if problem.constraint and problem.demand:
                salesman_idx = 0  # This is simplified - in practice you'd track which salesman
                current_demand = sum(problem.demand[city] for city in route)
                if current_demand + problem.demand[node] > problem.constraint[
                        salesman_idx]:
                    return False

        if problem.problem_type in [
                ProblemType.METRIC_CMTSPTW, ProblemType.GENERAL_CMTSPTW
        ]:
            # Check time window constraints (simplified)
            if problem.time_windows:
                # This is a simplified check - in practice you'd calculate actual arrival times
                return True

        return True

    def _find_best_insertion_cost(self, distance_matrix: np.ndarray,
                                  route: List[int], node: int) -> float:
        """Find the best insertion position for a node in a route"""
        if len(route) <= 1:
            return distance_matrix[route[0]][node] if route else 0

        best_cost = np.inf
        for i in range(len(route)):
            # Calculate cost of inserting at position i
            if i == 0:
                cost = distance_matrix[node][route[0]]
            elif i == len(route):
                cost = distance_matrix[route[-1]][node]
            else:
                cost = (distance_matrix[route[i - 1]][node] +
                        distance_matrix[node][route[i]] -
                        distance_matrix[route[i - 1]][route[i]])

            best_cost = min(best_cost, cost)

        return best_cost


class PortfolioSolver(BaseStandaloneSolver):
    """Solver for portfolio reallocation problems"""

    def __init__(self):
        super().__init__([ProblemType.PORTFOLIO])

    async def solve(self, problem: StandaloneTSPProblem) -> List[List[int]]:
        """Solve portfolio reallocation using greedy swaps"""
        if not self.is_valid_problem(problem):
            raise ValueError(
                f"Solver cannot handle problem type: {problem.problem_type}")

        # Simple greedy portfolio solver
        swaps = []
        portfolios = [
            portfolio.copy() for portfolio in problem.initial_portfolios
        ]

        # Try to satisfy constraints through swaps
        for iteration in range(100):  # Limit iterations
            best_swap = None
            best_improvement = 0

            # Find best swap
            for p1 in range(len(portfolios)):
                for p2 in range(len(portfolios)):
                    if p1 == p2:
                        continue

                    for subnet1 in range(len(portfolios[p1])):
                        for subnet2 in range(len(portfolios[p2])):
                            if portfolios[p1][subnet1] > 0 and portfolios[p2][
                                    subnet2] > 0:
                                # Try swapping
                                temp_p1 = portfolios[p1].copy()
                                temp_p2 = portfolios[p2].copy()

                                temp_p1[subnet1] -= 1
                                temp_p1[subnet2] += 1
                                temp_p2[subnet1] += 1
                                temp_p2[subnet2] -= 1

                                # Calculate improvement (simplified)
                                improvement = self._calculate_improvement(
                                    problem, portfolios, temp_p1, temp_p2, p1,
                                    p2)

                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_swap = (p1, p2, subnet1, subnet2)

            if best_swap:
                p1, p2, subnet1, subnet2 = best_swap
                portfolios[p1][subnet1] -= 1
                portfolios[p1][subnet2] += 1
                portfolios[p2][subnet1] += 1
                portfolios[p2][subnet2] -= 1
                swaps.append(
                    [p1, subnet1, subnet2,
                     1])  # [portfolio_idx, from_subnet, to_subnet, amount]
            else:
                break

        return swaps

    def _calculate_improvement(self, problem: StandaloneTSPProblem,
                               portfolios: List[List[int]], temp_p1: List[int],
                               temp_p2: List[int], p1: int, p2: int) -> float:
        """Calculate improvement from a potential swap"""
        # Simplified improvement calculation
        # In practice, this would consider constraint satisfaction and objective function
        return random.random()  # Placeholder


class SolverManager:
    """Manager for different solvers"""

    def __init__(self):
        self.solvers = {
            ProblemType.METRIC_TSP: NearestNeighborSolver(),
            ProblemType.GENERAL_TSP: NearestNeighborSolver(),
            ProblemType.METRIC_MTSP: MultiSalesmanSolver(),
            ProblemType.GENERAL_MTSP: MultiSalesmanSolver(),
            ProblemType.METRIC_CMTSP: MultiSalesmanSolver(),
            ProblemType.GENERAL_CMTSP: MultiSalesmanSolver(),
            ProblemType.METRIC_CMTSPTW: MultiSalesmanSolver(),
            ProblemType.GENERAL_CMTSPTW: MultiSalesmanSolver(),
            ProblemType.PORTFOLIO: PortfolioSolver()
        }

    async def solve(
        self, problem: StandaloneTSPProblem
    ) -> Union[List[int], List[List[int]], bool]:
        """Solve a problem using the appropriate solver"""
        solver = self.solvers.get(problem.problem_type)
        if not solver:
            raise ValueError(
                f"No solver available for problem type: {problem.problem_type}"
            )

        return await solver.solve(problem)


def calculate_solution_cost(
        problem: StandaloneTSPProblem,
        solution: Union[List[int], List[List[int]]]) -> float:
    """Calculate the cost of a solution"""
    if problem.problem_type == ProblemType.PORTFOLIO:
        # For portfolio problems, return number of swaps
        return len(solution) if isinstance(solution, list) else 0

    if isinstance(solution, list) and len(solution) > 0 and isinstance(
            solution[0], list):
        # Multi-salesman solution
        total_cost = 0
        distance_matrix = np.array(problem.edges)

        for route in solution:
            if len(route) > 1:
                for i in range(len(route) - 1):
                    total_cost += distance_matrix[route[i]][route[i + 1]]

        return total_cost
    else:
        # Single route solution
        if len(solution) <= 1:
            return 0

        distance_matrix = np.array(problem.edges)
        total_cost = 0

        for i in range(len(solution) - 1):
            total_cost += distance_matrix[solution[i]][solution[i + 1]]

        return total_cost


async def main():
    """Example usage of standalone solvers"""
    print("=== Standalone TSP Solver ===\n")

    from standalone_problem_generator import StandaloneProblemGenerator

    # Create problem generator and solver manager
    generator = StandaloneProblemGenerator()
    solver_manager = SolverManager()

    # Generate and solve different problems
    problems = [("Metric TSP", generator.generate_metric_tsp(10)),
                ("General TSP", generator.generate_general_tsp(10)),
                ("Multi-Salesman TSP", generator.generate_multi_tsp(8, 2)),
                ("Constrained Multi-Salesman TSP",
                 generator.generate_constrained_multi_tsp(8, 2)),
                ("Portfolio Problem",
                 generator.generate_portfolio_problem(3, 3))]

    for name, problem in problems:
        print(f"=== {name} ===")
        print(f"Problem type: {problem.problem_type}")
        print(f"Number of nodes: {problem.n_nodes}")

        try:
            # Solve the problem
            solution = await solver_manager.solve(problem)
            cost = calculate_solution_cost(problem, solution)

            print(f"Solution: {solution}")
            print(f"Cost: {cost:.2f}")

        except Exception as e:
            print(f"Error solving problem: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())
