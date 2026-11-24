# MINI-COURIER PLANNER
# Design and Analysis of Algorithms (ETCCDA302)
# Required algorithms: Merge Sort, 0/1 Knapsack, Dijkstra, Nearest Neighbour TSP

import heapq
import time
import csv
import random
from pathlib import Path

DEPOT = "Depot"
VEHICLE_CAPACITY = 12

# Sample parcel data: weight is in kg and deadline is in days.
PARCELS = [
    {"id": "P1", "destination": "A", "weight": 4, "value": 70, "deadline": 2},
    {"id": "P2", "destination": "B", "weight": 3, "value": 50, "deadline": 1},
    {"id": "P3", "destination": "C", "weight": 5, "value": 90, "deadline": 3},
    {"id": "P4", "destination": "D", "weight": 2, "value": 40, "deadline": 2},
    {"id": "P5", "destination": "E", "weight": 6, "value": 100, "deadline": 4},
    {"id": "P6", "destination": "F", "weight": 1, "value": 25, "deadline": 1},
]

# Undirected road graph. Each tuple contains (connected location, distance in km).
GRAPH = {
    "Depot": [("A", 4), ("B", 2), ("C", 7)],
    "A": [("Depot", 4), ("B", 1), ("D", 5)],
    "B": [("Depot", 2), ("A", 1), ("C", 3), ("E", 8)],
    "C": [("Depot", 7), ("B", 3), ("D", 2), ("F", 6)],
    "D": [("A", 5), ("C", 2), ("E", 2)],
    "E": [("B", 8), ("D", 2), ("F", 3)],
    "F": [("C", 6), ("E", 3)],
}


# 1. MERGE SORT
# Sort parcels by earliest delivery deadline.

def merge_sort(parcels):
    if len(parcels) <= 1:
        return parcels[:]

    middle = len(parcels) // 2
    left = merge_sort(parcels[:middle])
    right = merge_sort(parcels[middle:])

    sorted_list = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        left_key = (left[i]["deadline"], left[i]["id"])
        right_key = (right[j]["deadline"], right[j]["id"])

        if left_key <= right_key:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1

    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list


# 2. 0/1 KNAPSACK USING DYNAMIC PROGRAMMING
# Select parcels with maximum total value without exceeding vehicle capacity.

def knapsack(parcels, capacity):
    n = len(parcels)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight = parcels[i - 1]["weight"]
        value = parcels[i - 1]["value"]

        for current_capacity in range(capacity + 1):
            dp[i][current_capacity] = dp[i - 1][current_capacity]

            if weight <= current_capacity:
                value_if_selected = value + dp[i - 1][current_capacity - weight]
                dp[i][current_capacity] = max(
                    dp[i][current_capacity], value_if_selected
                )

    selected_parcels = []
    remaining_capacity = capacity

    for i in range(n, 0, -1):
        if dp[i][remaining_capacity] != dp[i - 1][remaining_capacity]:
            parcel = parcels[i - 1]
            selected_parcels.append(parcel)
            remaining_capacity -= parcel["weight"]

    selected_parcels.reverse()

    total_weight = sum(parcel["weight"] for parcel in selected_parcels)
    total_value = sum(parcel["value"] for parcel in selected_parcels)

    return selected_parcels, total_weight, total_value


# 3. DIJKSTRA'S ALGORITHM
# Find shortest distances from one starting location to all other locations.

def dijkstra(graph, start):
    distances = {location: float("inf") for location in graph}
    previous = {location: None for location in graph}
    distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_location = heapq.heappop(priority_queue)

        if current_distance > distances[current_location]:
            continue

        for neighbour, road_distance in graph[current_location]:
            new_distance = current_distance + road_distance

            if new_distance < distances[neighbour]:
                distances[neighbour] = new_distance
                previous[neighbour] = current_location
                heapq.heappush(priority_queue, (new_distance, neighbour))

    return distances, previous


def build_path(previous, start, destination):
    path = []
    current = destination

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    if not path or path[0] != start:
        return []

    return path


def create_distance_matrix(graph, locations):
    distance_matrix = {}
    shortest_paths = {}

    for source in locations:
        distances, previous = dijkstra(graph, source)
        distance_matrix[source] = {}
        shortest_paths[source] = {}

        for destination in locations:
            distance_matrix[source][destination] = distances[destination]
            shortest_paths[source][destination] = build_path(
                previous, source, destination
            )

    return distance_matrix, shortest_paths


# 4. NEAREST NEIGHBOUR TSP APPROXIMATION
# Visit the nearest unvisited delivery location and finally return to the depot.

def nearest_neighbour(locations, distance_matrix, start):
    unvisited = set(locations)
    unvisited.remove(start)

    route = [start]
    total_distance = 0
    current = start

    while unvisited:
        next_location = min(
            unvisited,
            key=lambda location: distance_matrix[current][location]
        )

        total_distance += distance_matrix[current][next_location]
        route.append(next_location)
        unvisited.remove(next_location)
        current = next_location

    total_distance += distance_matrix[current][start]
    route.append(start)

    return route, total_distance


def expand_route(route, shortest_paths):
    complete_path = []

    for index in range(len(route) - 1):
        leg = shortest_paths[route[index]][route[index + 1]]

        if index == 0:
            complete_path.extend(leg)
        else:
            complete_path.extend(leg[1:])

    return complete_path


# 5. PERFORMANCE TESTING
# Measure Merge Sort and Knapsack for increasing input sizes.

def performance_test():
    results = []

    for size in [10, 50, 100, 200]:
        test_parcels = []

        for index in range(size):
            test_parcels.append({
                "id": "T" + str(index),
                "destination": "A",
                "weight": random.randint(1, 10),
                "value": random.randint(10, 100),
                "deadline": random.randint(1, 30),
            })

        start_time = time.perf_counter()
        merge_sort(test_parcels)
        merge_sort_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        knapsack(test_parcels, 50)
        knapsack_time = time.perf_counter() - start_time

        results.append([size, merge_sort_time, knapsack_time])

    output_file = Path(__file__).with_name("performance_results.csv")

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Input Size",
            "Merge Sort Time (seconds)",
            "Knapsack Time (seconds)"
        ])
        writer.writerows(results)

    print("\nPERFORMANCE RESULTS")
    print("Input Size | Merge Sort (seconds) | Knapsack (seconds)")

    for row in results:
        print(
            row[0], "|",
            format(row[1], ".8f"), "|",
            format(row[2], ".8f")
        )

    print("Performance results saved to performance_results.csv")


# MAIN PROGRAM

def main():
    print("=" * 60)
    print("                 MINI-COURIER PLANNER")
    print("=" * 60)

    # Step 1: Sort parcels by delivery deadline.
    sorted_parcels = merge_sort(PARCELS)

    print("\n1. PARCELS SORTED BY DEADLINE")
    for parcel in sorted_parcels:
        print(
            parcel["id"],
            "| Destination:", parcel["destination"],
            "| Deadline:", parcel["deadline"],
            "| Weight:", parcel["weight"], "kg",
            "| Value:", parcel["value"]
        )

    # Step 2: Select the best parcels within the vehicle capacity.
    selected, total_weight, total_value = knapsack(
        sorted_parcels, VEHICLE_CAPACITY
    )

    print("\n2. PARCELS SELECTED USING 0/1 KNAPSACK")
    for parcel in selected:
        print(parcel["id"], "->", parcel["destination"])

    print("Vehicle capacity:", VEHICLE_CAPACITY, "kg")
    print("Selected weight:", total_weight, "kg")
    print("Total parcel value:", total_value)

    # Step 3: Prepare the depot and selected delivery locations.
    destinations = sorted(set(
        parcel["destination"] for parcel in selected
    ))
    locations = [DEPOT] + destinations

    # Step 4: Calculate shortest paths using Dijkstra.
    distance_matrix, shortest_paths = create_distance_matrix(
        GRAPH, locations
    )

    print("\n3. DIJKSTRA SHORTEST-PATH DISTANCES")
    for source in locations:
        for destination in locations:
            if source != destination:
                print(
                    source, "to", destination, "=",
                    distance_matrix[source][destination], "km"
                )

    # Step 5: Create the approximate delivery route.
    route, route_distance = nearest_neighbour(
        locations, distance_matrix, DEPOT
    )
    complete_path = expand_route(route, shortest_paths)

    print("\n4. FINAL DELIVERY ROUTE")
    print("Stop order:", " -> ".join(route))
    print("Road path:", " -> ".join(complete_path))
    print("Total route distance:", route_distance, "km")

    # Step 6: Compare execution times.
    print("\n5. PERFORMANCE TESTING")
    performance_test()

    print("\nPROJECT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
