#!/usr/bin/env python3
"""
Final TSP Example with Real Data
Comprehensive example showing how to use the TSP system with your .npz files
"""

from tsp_with_real_data import (RealDataProblemGenerator, RealDataTSPSolver,
                                calculate_solution_cost, ProblemType)


def demonstrate_basic_usage():
    """Demonstrate basic usage of the TSP system"""
    print("=== Basic Usage Example ===")

    # Create generator and solver
    generator = RealDataProblemGenerator()
    solver = RealDataTSPSolver()

    # Show available datasets
    datasets = generator.get_available_datasets()
    print(f"Available datasets: {datasets}")

    # Generate a simple TSP problem
    problem = generator.generate_metric_tsp("Asia_MSB", 8)
    print(f"\nGenerated problem: {problem.get_info()}")

    # Solve the problem
    solution = solver.solve(problem)
    cost = calculate_solution_cost(problem, solution)

    print(f"Solution: {solution}")
    print(f"Total distance: {cost:.2f} km")
    print()


def demonstrate_multi_salesman():
    """Demonstrate multi-salesman TSP"""
    print("=== Multi-Salesman TSP Example ===")

    generator = RealDataProblemGenerator()
    solver = RealDataTSPSolver()

    # Generate multi-salesman problem
    problem = generator.generate_multi_tsp("USA_POI", 15, 3)
    print(f"Generated problem: {problem.get_info()}")

    # Solve the problem
    solution = solver.solve(problem)
    cost = calculate_solution_cost(problem, solution)

    print(f"Solution: {len(solution)} routes")
    for i, route in enumerate(solution):
        print(f"  Salesman {i+1}: {route} (visits {len(route)-1} cities)")
    print(f"Total distance: {cost:.2f} km")
    print()


def demonstrate_different_datasets():
    """Demonstrate different datasets"""
    print("=== Different Datasets Example ===")

    generator = RealDataProblemGenerator()
    solver = RealDataTSPSolver()

    datasets = generator.get_available_datasets()

    for dataset in datasets:
        print(f"\n--- {dataset} Dataset ---")

        # Generate problem
        problem = generator.generate_metric_tsp(dataset, 10)
        info = problem.get_info()

        print(f"Geographic Area: {info['Geographic Area']}")
        print(
            f"Coordinate Range: {info['Longitude Range']}, {info['Latitude Range']}"
        )

        # Solve
        solution = solver.solve(problem)
        cost = calculate_solution_cost(problem, solution)

        print(f"Solution: {solution}")
        print(f"Total distance: {cost:.2f} km")

        # Show some cities
        print("Sample cities:")
        for i, city in enumerate(problem.cities[:3]):
            print(
                f"  City {city.id}: {city.longitude:.2f}°E, {city.latitude:.2f}°N"
            )


def demonstrate_problem_sizes():
    """Demonstrate different problem sizes"""
    print("\n=== Problem Size Comparison ===")

    generator = RealDataProblemGenerator()
    solver = RealDataTSPSolver()

    sizes = [5, 10, 15, 20]

    print("Size | Distance (km) | Route Length")
    print("-" * 40)

    for size in sizes:
        problem = generator.generate_metric_tsp("Asia_MSB", size)
        solution = solver.solve(problem)
        cost = calculate_solution_cost(problem, solution)

        print(f"{size:4d} | {cost:12.2f} | {len(solution):12d}")


def demonstrate_custom_problem():
    """Demonstrate creating a custom problem"""
    print("\n=== Custom Problem Example ===")

    from tsp_with_real_data import RealDataTSPProblem, CityData, ProblemType

    # Create custom cities
    custom_cities = [
        CityData(0, 139.6917, 35.6895),  # Tokyo
        CityData(1, 121.4737, 31.2304),  # Shanghai
        CityData(2, 114.1694, 22.3193),  # Hong Kong
        CityData(3, 103.8198, 1.3521),  # Singapore
        CityData(4, 126.9780, 37.5665),  # Seoul
    ]

    # Create problem
    problem = RealDataTSPProblem(ProblemType.METRIC_TSP, custom_cities)

    print("Custom Asian cities TSP:")
    print("Cities:")
    for city in problem.cities:
        print(f"  {city.id}: {city.longitude:.2f}°E, {city.latitude:.2f}°N")

    # Solve
    solver = RealDataTSPSolver()
    solution = solver.solve(problem)
    cost = calculate_solution_cost(problem, solution)

    print(f"Solution: {solution}")
    print(f"Total distance: {cost:.2f} km")


def demonstrate_benchmark():
    """Demonstrate benchmarking different approaches"""
    print("\n=== Benchmarking Example ===")

    generator = RealDataProblemGenerator()
    solver = RealDataTSPSolver()

    # Test different configurations
    configs = [
        ("Asia_MSB", 12, 1),
        ("Asia_MSB", 12, 2),
        ("USA_POI", 12, 1),
        ("USA_POI", 12, 2),
        ("World_TSP", 12, 1),
        ("World_TSP", 12, 2),
    ]

    print("Dataset     | Salesmen | Distance (km) | Cities per Route")
    print("-" * 60)

    for dataset, n_cities, n_salesmen in configs:
        if n_salesmen == 1:
            problem = generator.generate_metric_tsp(dataset, n_cities)
        else:
            problem = generator.generate_multi_tsp(dataset, n_cities,
                                                   n_salesmen)

        solution = solver.solve(problem)
        cost = calculate_solution_cost(problem, solution)

        if isinstance(solution, list) and len(solution) > 0 and isinstance(
                solution[0], list):
            cities_per_route = [len(route) - 1 for route in solution]
            cities_str = str(cities_per_route)
        else:
            cities_str = str(len(solution) - 1)

        print(f"{dataset:11s} | {n_salesmen:8d} | {cost:12.2f} | {cities_str}")


def main():
    """Main function demonstrating all features"""
    print("TSP System with Real-World Data - Complete Example")
    print("=" * 60)

    demonstrate_basic_usage()
    demonstrate_multi_salesman()
    demonstrate_different_datasets()
    demonstrate_problem_sizes()
    demonstrate_custom_problem()
    demonstrate_benchmark()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nTo use this system in your own code:")
    print("1. Import the necessary classes")
    print("2. Create a RealDataProblemGenerator")
    print("3. Generate problems using your .npz files")
    print("4. Solve using RealDataTSPSolver")
    print("5. Calculate costs using calculate_solution_cost")


if __name__ == "__main__":
    main()
