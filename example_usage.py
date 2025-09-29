#!/usr/bin/env python3
"""
Example usage of the standalone TSP system
Shows how to create and solve various TSP problems without Bittensor
"""

from simple_tsp_generator import (create_problem_generator, TSPSolver,
                                  calculate_solution_cost, ProblemType)


def example_basic_tsp():
    """Example: Basic TSP problem"""
    print("=== Basic TSP Example ===")

    generator = create_problem_generator()
    solver = TSPSolver()

    # Create a metric TSP problem
    problem = generator.generate_metric_tsp(n_nodes=15)
    print(f"Created {problem.problem_type.value} with {problem.n_nodes} nodes")

    # Show some coordinates
    if problem.nodes:
        print("First 3 node coordinates:")
        for i, node in enumerate(problem.nodes[:3]):
            print(f"  Node {i}: ({node[0]:.1f}, {node[1]:.1f})")

    # Solve the problem
    solution = solver.solve(problem)
    cost = calculate_solution_cost(problem, solution)

    print(f"Solution route: {solution}")
    print(f"Total distance: {cost:.2f}")
    print()


def example_multi_salesman():
    """Example: Multi-salesman TSP problem"""
    print("=== Multi-Salesman TSP Example ===")

    generator = create_problem_generator()
    solver = TSPSolver()

    # Create a multi-salesman problem
    problem = generator.generate_multi_tsp(n_nodes=20, n_salesmen=3)
    print(
        f"Created {problem.problem_type.value} with {problem.n_salesmen} salesmen"
    )
    print(f"Depots: {problem.depots}")

    # Solve the problem
    solution = solver.solve(problem)
    cost = calculate_solution_cost(problem, solution)

    print(f"Solution: {len(solution)} routes")
    for i, route in enumerate(solution):
        print(f"  Salesman {i+1}: {route} (visits {len(route)-1} cities)")

    print(f"Total distance: {cost:.2f}")
    print()


def example_comparison():
    """Example: Compare different problem types"""
    print("=== Problem Type Comparison ===")

    generator = create_problem_generator()
    solver = TSPSolver()

    problems = [("Metric TSP", generator.generate_metric_tsp(12)),
                ("General TSP", generator.generate_general_tsp(12)),
                ("Multi-Salesman TSP", generator.generate_multi_tsp(12, 2))]

    for name, problem in problems:
        solution = solver.solve(problem)
        cost = calculate_solution_cost(problem, solution)

        print(f"{name}:")
        print(f"  Cost: {cost:.2f}")
        if isinstance(solution, list) and len(solution) > 0 and isinstance(
                solution[0], list):
            print(f"  Routes: {len(solution)}")
        else:
            print(f"  Route length: {len(solution)}")
    print()


def example_custom_problem():
    """Example: Create a custom problem with specific coordinates"""
    print("=== Custom Problem Example ===")

    # Create a custom problem with known coordinates
    problem = ProblemType.METRIC_TSP
    n_nodes = 5

    # Define specific city coordinates
    custom_nodes = [
        [0, 0],  # City 0
        [10, 0],  # City 1
        [10, 10],  # City 2
        [0, 10],  # City 3
        [5, 5]  # City 4
    ]

    # Create problem manually
    from simple_tsp_generator import TSPProblem
    problem = TSPProblem(problem_type=ProblemType.METRIC_TSP,
                         n_nodes=n_nodes,
                         nodes=custom_nodes)

    print("Custom problem with coordinates:")
    for i, node in enumerate(problem.nodes):
        print(f"  City {i}: ({node[0]}, {node[1]})")

    # Solve
    solver = TSPSolver()
    solution = solver.solve(problem)
    cost = calculate_solution_cost(problem, solution)

    print(f"Optimal route: {solution}")
    print(f"Total distance: {cost:.2f}")
    print()


def example_benchmark():
    """Example: Benchmark different problem sizes"""
    print("=== Benchmark Example ===")

    generator = create_problem_generator()
    solver = TSPSolver()

    problem_sizes = [5, 10, 15, 20]

    print("Problem Size | Solution Cost | Route Length")
    print("-" * 40)

    for size in problem_sizes:
        problem = generator.generate_metric_tsp(size)
        solution = solver.solve(problem)
        cost = calculate_solution_cost(problem, solution)

        print(f"{size:11d} | {cost:12.2f} | {len(solution):12d}")

    print()


def main():
    """Run all examples"""
    print("Standalone TSP System - Usage Examples")
    print("=" * 50)

    example_basic_tsp()
    example_multi_salesman()
    example_comparison()
    example_custom_problem()
    example_benchmark()

    print("All examples completed!")


if __name__ == "__main__":
    main()
