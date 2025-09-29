#!/usr/bin/env python3
"""
Test script for standalone TSP problem generation and solving
Demonstrates the system without Bittensor dependencies
"""

import asyncio
import time
import numpy as np
from standalone_problem_generator import StandaloneProblemGenerator, ProblemType
from standalone_solver import SolverManager, calculate_solution_cost


class TSPBenchmark:
    """Benchmark different TSP solvers and problem types"""

    def __init__(self):
        self.generator = StandaloneProblemGenerator()
        self.solver_manager = SolverManager()
        self.results = []

    async def benchmark_problem_type(self,
                                     problem_type: str,
                                     generator_func,
                                     solver_name: str,
                                     n_trials: int = 5,
                                     **kwargs):
        """Benchmark a specific problem type"""
        print(f"\n=== Benchmarking {problem_type} ===")

        costs = []
        solve_times = []

        for trial in range(n_trials):
            # Generate problem
            problem = generator_func(**kwargs)

            # Solve problem
            start_time = time.time()
            try:
                solution = await self.solver_manager.solve(problem)
                solve_time = time.time() - start_time

                # Calculate cost
                cost = calculate_solution_cost(problem, solution)

                costs.append(cost)
                solve_times.append(solve_time)

                print(
                    f"Trial {trial + 1}: Cost = {cost:.2f}, Time = {solve_time:.3f}s"
                )

            except Exception as e:
                print(f"Trial {trial + 1}: Error - {e}")
                costs.append(float('inf'))
                solve_times.append(float('inf'))

        # Calculate statistics
        valid_costs = [c for c in costs if c != float('inf')]
        valid_times = [t for t in solve_times if t != float('inf')]

        if valid_costs:
            avg_cost = np.mean(valid_costs)
            std_cost = np.std(valid_costs)
            avg_time = np.mean(valid_times)
            std_time = np.std(valid_times)

            print(f"Average Cost: {avg_cost:.2f} ± {std_cost:.2f}")
            print(f"Average Time: {avg_time:.3f}s ± {std_time:.3f}s")

            self.results.append({
                'problem_type': problem_type,
                'solver': solver_name,
                'avg_cost': avg_cost,
                'std_cost': std_cost,
                'avg_time': avg_time,
                'std_time': std_time,
                'success_rate': len(valid_costs) / n_trials
            })
        else:
            print("No successful solutions found!")
            self.results.append({
                'problem_type': problem_type,
                'solver': solver_name,
                'avg_cost': float('inf'),
                'std_cost': 0,
                'avg_time': float('inf'),
                'std_time': 0,
                'success_rate': 0
            })

    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for result in self.results:
            print(f"\n{result['problem_type']} ({result['solver']}):")
            print(f"  Success Rate: {result['success_rate']:.1%}")
            if result['success_rate'] > 0:
                print(
                    f"  Average Cost: {result['avg_cost']:.2f} ± {result['std_cost']:.2f}"
                )
                print(
                    f"  Average Time: {result['avg_time']:.3f}s ± {result['std_time']:.3f}s"
                )


async def test_problem_generation():
    """Test problem generation capabilities"""
    print("=== Testing Problem Generation ===")

    generator = StandaloneProblemGenerator()

    # Test different problem types
    problems = [("Small Metric TSP", generator.generate_metric_tsp(10)),
                ("Large Metric TSP", generator.generate_metric_tsp(50)),
                ("General TSP", generator.generate_general_tsp(20)),
                ("Multi-Salesman TSP", generator.generate_multi_tsp(15, 3)),
                ("Constrained Multi-Salesman TSP",
                 generator.generate_constrained_multi_tsp(15, 3)),
                ("Time-Window TSP", generator.generate_time_window_tsp(15, 3)),
                ("Portfolio Problem",
                 generator.generate_portfolio_problem(5, 5))]

    for name, problem in problems:
        print(f"\n{name}:")
        info = problem.get_info(verbosity=2)
        for key, value in info.items():
            print(f"  {key}: {value}")

        if problem.edges:
            print(
                f"  Edge matrix shape: {len(problem.edges)}x{len(problem.edges[0])}"
            )

        if problem.nodes:
            print(f"  Node coordinates: {len(problem.nodes)} nodes")


async def test_solver_performance():
    """Test solver performance on different problem sizes"""
    print("\n=== Testing Solver Performance ===")

    benchmark = TSPBenchmark()

    # Test different problem sizes
    problem_sizes = [10, 20, 50]

    for size in problem_sizes:
        await benchmark.benchmark_problem_type(
            f"Metric TSP (n={size})",
            benchmark.generator.generate_metric_tsp,
            "NearestNeighbor",
            n_trials=3,
            n_nodes=size)

    # Test multi-salesman problems
    await benchmark.benchmark_problem_type(
        "Multi-Salesman TSP",
        benchmark.generator.generate_multi_tsp,
        "MultiSalesman",
        n_trials=3,
        n_nodes=15,
        n_salesmen=3)

    # Test portfolio problems
    await benchmark.benchmark_problem_type(
        "Portfolio Problem",
        benchmark.generator.generate_portfolio_problem,
        "Portfolio",
        n_trials=3,
        n_portfolio=5,
        n_subnets=5)

    benchmark.print_summary()


async def test_solution_quality():
    """Test solution quality and validation"""
    print("\n=== Testing Solution Quality ===")

    generator = StandaloneProblemGenerator()
    solver_manager = SolverManager()

    # Generate a medium-sized problem
    problem = generator.generate_metric_tsp(20)
    print(f"Generated problem with {problem.n_nodes} nodes")

    # Solve multiple times to check consistency
    solutions = []
    costs = []

    for i in range(5):
        solution = await solver_manager.solve(problem)
        cost = calculate_solution_cost(problem, solution)
        solutions.append(solution)
        costs.append(cost)
        print(
            f"Solution {i+1}: Cost = {cost:.2f}, Route length = {len(solution)}"
        )

    # Analyze solution consistency
    if costs:
        avg_cost = np.mean(costs)
        std_cost = np.std(costs)
        print(f"\nCost Statistics:")
        print(f"  Average: {avg_cost:.2f}")
        print(f"  Std Dev: {std_cost:.2f}")
        print(f"  Min: {min(costs):.2f}")
        print(f"  Max: {max(costs):.2f}")

    # Check if solutions are valid
    print(f"\nSolution Validation:")
    for i, solution in enumerate(solutions):
        is_valid = len(
            solution) == problem.n_nodes + 1  # +1 for return to start
        print(f"  Solution {i+1}: Valid = {is_valid}")


async def demo_advanced_features():
    """Demonstrate advanced features"""
    print("\n=== Advanced Features Demo ===")

    generator = StandaloneProblemGenerator()
    solver_manager = SolverManager()

    # Demo time-window constraints
    print("1. Time-Window Constrained TSP:")
    tw_problem = generator.generate_time_window_tsp(10, 2)
    print(
        f"   Generated problem with {len(tw_problem.time_windows)} time windows"
    )
    print(f"   Time windows: {tw_problem.time_windows[:3]}...")  # Show first 3

    try:
        tw_solution = await solver_manager.solve(tw_problem)
        print(f"   Solution: {len(tw_solution)} routes")
        for i, route in enumerate(tw_solution):
            print(f"   Route {i+1}: {route}")
    except Exception as e:
        print(f"   Error: {e}")

    # Demo portfolio optimization
    print("\n2. Portfolio Optimization:")
    portfolio_problem = generator.generate_portfolio_problem(3, 4)
    print(
        f"   Generated portfolio problem with {portfolio_problem.n_portfolio} portfolios"
    )
    print(f"   Constraint types: {portfolio_problem.constraint_types}")
    print(f"   Constraint values: {portfolio_problem.constraint_values}")

    try:
        portfolio_solution = await solver_manager.solve(portfolio_problem)
        print(f"   Solution: {len(portfolio_solution)} swaps")
        for i, swap in enumerate(portfolio_solution[:3]):  # Show first 3 swaps
            print(f"   Swap {i+1}: {swap}")
    except Exception as e:
        print(f"   Error: {e}")


async def main():
    """Main test function"""
    print("Standalone TSP System Test Suite")
    print("=" * 50)

    # Run all tests
    await test_problem_generation()
    await test_solver_performance()
    await test_solution_quality()
    await demo_advanced_features()

    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
