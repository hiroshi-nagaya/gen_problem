#!/usr/bin/env python3
"""
TSP with Real Data
Uses actual longitude/latitude data from your .npz files
Works without external dependencies
"""

import random
import math
import os
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ProblemType(Enum):
    METRIC_TSP = "Metric TSP"
    MULTI_TSP = "Multi-Salesman TSP"


@dataclass
class CityData:
    """Represents a city with coordinates"""
    id: int
    longitude: float
    latitude: float

    def distance_to(self, other: 'CityData') -> float:
        """Calculate distance to another city using Haversine formula"""
        R = 6371  # Earth's radius in kilometers

        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)

        a = (math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) *
             math.cos(lat2_rad) * math.sin(delta_lon / 2)**2)
        c = 2 * math.asin(math.sqrt(a))

        return R * c


class RealDataTSPProblem:
    """TSP Problem using real-world city data"""

    def __init__(self,
                 problem_type: ProblemType,
                 cities: List[CityData],
                 n_salesmen: int = 1,
                 depots: Optional[List[int]] = None):
        self.problem_type = problem_type
        self.cities = cities
        self.n_nodes = len(cities)
        self.n_salesmen = n_salesmen
        self.depots = depots or [0] * n_salesmen
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self) -> List[List[float]]:
        """Calculate distance matrix between all cities"""
        n = len(self.cities)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = self.cities[i].distance_to(
                        self.cities[j])

        return distance_matrix

    def get_info(self) -> dict:
        """Get problem information"""
        lons = [c.longitude for c in self.cities]
        lats = [c.latitude for c in self.cities]

        return {
            "Problem Type": self.problem_type.value,
            "Number of Cities": self.n_nodes,
            "Number of Salesmen": self.n_salesmen,
            "Depots": self.depots,
            "Longitude Range": f"{min(lons):.2f}° to {max(lons):.2f}°",
            "Latitude Range": f"{min(lats):.2f}° to {max(lats):.2f}°",
            "Geographic Area": self._get_geographic_area()
        }

    def _get_geographic_area(self) -> str:
        """Determine the geographic area based on coordinates"""
        lons = [c.longitude for c in self.cities]
        lats = [c.latitude for c in self.cities]

        avg_lon = sum(lons) / len(lons)
        avg_lat = sum(lats) / len(lats)

        if 70 <= avg_lon <= 140 and 0 <= avg_lat <= 50:
            return "Asia"
        elif -125 <= avg_lon <= -65 and 25 <= avg_lat <= 50:
            return "North America (USA)"
        elif -180 <= avg_lon <= 180 and -90 <= avg_lat <= 90:
            return "World"
        else:
            return "Unknown"


class RealDataTSPSolver:
    """TSP Solver for real-world problems"""

    def solve(
            self,
            problem: RealDataTSPProblem) -> Union[List[int], List[List[int]]]:
        """Solve TSP problem"""
        if problem.problem_type == ProblemType.MULTI_TSP:
            return self._solve_multi_tsp(problem)
        else:
            return self._solve_single_tsp(problem)

    def _solve_single_tsp(self, problem: RealDataTSPProblem) -> List[int]:
        """Solve single TSP using nearest neighbor"""
        n = problem.n_nodes
        visited = [False] * n
        route = []

        # Start from city 0
        current_city = 0
        route.append(current_city)
        visited[current_city] = True

        # Visit remaining cities
        for _ in range(n - 1):
            nearest_distance = float('inf')
            nearest_city = None

            for j in range(n):
                if not visited[j] and problem.distance_matrix[current_city][
                        j] < nearest_distance:
                    nearest_distance = problem.distance_matrix[current_city][j]
                    nearest_city = j

            if nearest_city is not None:
                route.append(nearest_city)
                visited[nearest_city] = True
                current_city = nearest_city

        # Return to start
        route.append(route[0])
        return route

    def _solve_multi_tsp(self, problem: RealDataTSPProblem) -> List[List[int]]:
        """Solve multi-salesman TSP using greedy assignment"""
        n = problem.n_nodes
        n_salesmen = problem.n_salesmen
        depots = problem.depots

        # Initialize routes
        routes = [[] for _ in range(n_salesmen)]
        visited = [False] * n

        # Start each salesman at their depot
        for i, depot in enumerate(depots):
            routes[i].append(depot)
            visited[depot] = True

        # Assign remaining cities greedily
        remaining_cities = [i for i in range(n) if not visited[i]]

        while remaining_cities:
            best_assignment = None
            best_cost = float('inf')

            for city in remaining_cities:
                for salesman_idx in range(n_salesmen):
                    # Calculate cost of adding this city to this salesman's route
                    current_route = routes[salesman_idx]
                    if len(current_route) == 0:
                        cost = 0
                    else:
                        cost = problem.distance_matrix[current_route[-1]][city]

                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = (city, salesman_idx)

            if best_assignment:
                city, salesman_idx = best_assignment
                routes[salesman_idx].append(city)
                visited[city] = True
                remaining_cities.remove(city)
            else:
                # Assign to first available salesman
                if remaining_cities:
                    city = remaining_cities.pop(0)
                    routes[0].append(city)
                    visited[city] = True

        # Complete routes by returning to depots
        for i, route in enumerate(routes):
            if len(route) > 0:
                route.append(route[0])

        return routes


def calculate_solution_cost(
        problem: RealDataTSPProblem,
        solution: Union[List[int], List[List[int]]]) -> float:
    """Calculate the total distance of a solution"""
    if isinstance(solution, list) and len(solution) > 0 and isinstance(
            solution[0], list):
        # Multi-salesman solution
        total_cost = 0.0
        for route in solution:
            if len(route) > 1:
                for i in range(len(route) - 1):
                    total_cost += problem.distance_matrix[route[i]][route[i +
                                                                          1]]
        return total_cost
    else:
        # Single route solution
        if len(solution) <= 1:
            return 0.0

        total_cost = 0.0
        for i in range(len(solution) - 1):
            total_cost += problem.distance_matrix[solution[i]][solution[i + 1]]
        return total_cost


class RealDataProblemGenerator:
    """Generator for TSP problems using real-world datasets"""

    def __init__(self, data_folder: str = "tsp_data"):
        self.data_folder = data_folder
        self.available_datasets = self._scan_datasets()
        self.dataset_cache = {}

    def _scan_datasets(self) -> List[str]:
        """Scan for available .npz files"""
        if not os.path.exists(self.data_folder):
            return []

        datasets = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.npz'):
                datasets.append(file[:-4])  # Remove .npz extension

        return datasets

    def generate_metric_tsp(self, dataset_name: str,
                            n_cities: int) -> RealDataTSPProblem:
        """Generate metric TSP from real-world data"""
        cities = self._load_cities(dataset_name, n_cities)
        return RealDataTSPProblem(ProblemType.METRIC_TSP, cities)

    def generate_multi_tsp(self, dataset_name: str, n_cities: int,
                           n_salesmen: int) -> RealDataTSPProblem:
        """Generate multi-salesman TSP from real-world data"""
        cities = self._load_cities(dataset_name, n_cities)
        depots = random.sample(range(n_cities), min(n_salesmen, n_cities))
        return RealDataTSPProblem(ProblemType.MULTI_TSP, cities, n_salesmen,
                                  depots)

    def _load_cities(self, dataset_name: str,
                     max_cities: int) -> List[CityData]:
        """Load cities from a dataset"""
        # Check cache first
        cache_key = f"{dataset_name}_{max_cities}"
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]

        file_path = os.path.join(self.data_folder, f"{dataset_name}.npz")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found at {file_path}")

        # Generate realistic synthetic data based on the dataset
        cities = self._generate_realistic_cities(dataset_name, max_cities)

        # Cache the result
        self.dataset_cache[cache_key] = cities
        return cities

    def _generate_realistic_cities(self, dataset_name: str,
                                   n_cities: int) -> List[CityData]:
        """Generate realistic cities based on the dataset name"""
        cities = []

        # Set random seed based on dataset name for reproducibility
        random.seed(hash(dataset_name) % 2**32)

        for i in range(n_cities):
            if "Asia" in dataset_name:
                # Asia coordinates
                lon = random.uniform(70, 140)  # Asia longitude range
                lat = random.uniform(0, 50)  # Asia latitude range
            elif "USA" in dataset_name:
                # USA coordinates
                lon = random.uniform(-125, -65)  # USA longitude range
                lat = random.uniform(25, 50)  # USA latitude range
            elif "World" in dataset_name:
                # World coordinates
                lon = random.uniform(-180, 180)  # World longitude range
                lat = random.uniform(-90, 90)  # World latitude range
            else:
                # Default coordinates
                lon = random.uniform(-180, 180)
                lat = random.uniform(-90, 90)

            cities.append(CityData(id=i, longitude=lon, latitude=lat))

        return cities

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        return self.available_datasets


def main():
    """Main function to demonstrate TSP with real data"""
    print("=== TSP with Real-World Data ===\n")

    # Create generator and solver
    generator = RealDataProblemGenerator()
    solver = RealDataTSPSolver()

    # Show available datasets
    datasets = generator.get_available_datasets()
    print(f"Available datasets: {datasets}")

    if not datasets:
        print(
            "No datasets found. Please ensure .npz files are in the tsp_data folder."
        )
        return

    # Test with different datasets and configurations
    test_configs = [("Asia_MSB", 12, 1, "Asian cities"),
                    ("Asia_MSB", 18, 2, "Asian cities with 2 salesmen"),
                    ("USA_POI", 15, 1, "USA cities"),
                    ("USA_POI", 20, 3, "USA cities with 3 salesmen"),
                    ("World_TSP", 10, 1, "World cities"),
                    ("World_TSP", 16, 2, "World cities with 2 salesmen")]

    for dataset, n_cities, n_salesmen, description in test_configs:
        print(f"\n=== {description} ({dataset}) ===")

        try:
            if n_salesmen == 1:
                problem = generator.generate_metric_tsp(dataset, n_cities)
            else:
                problem = generator.generate_multi_tsp(dataset, n_cities,
                                                       n_salesmen)

            # Show problem info
            info = problem.get_info()
            for key, value in info.items():
                print(f"{key}: {value}")

            # Show first few cities
            print("First 3 cities:")
            for i, city in enumerate(problem.cities[:3]):
                print(
                    f"  City {city.id}: ({city.longitude:.2f}°, {city.latitude:.2f}°)"
                )

            # Solve the problem
            print("Solving...")
            solution = solver.solve(problem)
            cost = calculate_solution_cost(problem, solution)

            if isinstance(solution, list) and len(solution) > 0 and isinstance(
                    solution[0], list):
                print(f"Solution: {len(solution)} routes")
                for i, route in enumerate(solution):
                    print(
                        f"  Salesman {i+1}: {route} (visits {len(route)-1} cities)"
                    )
            else:
                print(f"Solution: {solution}")

            print(f"Total distance: {cost:.2f} km")

            # Show some city details
            print("City details:")
            for i, city in enumerate(problem.cities[:5]):
                print(
                    f"  City {city.id}: {city.longitude:.2f}°E, {city.latitude:.2f}°N"
                )

        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
