"""
Data Generator for Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

This module generates random problem instances including:
- Geographic customer locations around a central depot
- Distance and travel time matrices
- Customer demands
- Time window constraints for each customer
"""

import numpy as np
from geopy.distance import geodesic
from geopy import Point as GeoPoint
from geopy.distance import distance


class CVRPTWDataGenerator:
    """
    Generates random CVRPTW problem instances with geographic context.
    """

    def __init__(self, config):
        """
        Initialize data generator with problem configuration.

        Args:
            config (dict): Configuration dictionary containing problem parameters
        """
        np.random.seed(config.get('random_seed', 42))
        self.config = config

        # Generate problem data
        self._generate_customer_locations()
        self._compute_distance_matrix()
        self._generate_customer_demands()
        self._generate_time_windows()
        self._compute_travel_times()

    def _generate_customer_locations(self):
        """
        Generate random customer locations around the depot using polar coordinates.
        Customers are uniformly distributed within a circle of given radius.
        """
        depot_location = self.config['depot_location']
        radius_miles = self.config['radius_miles']
        num_customers = self.config['num_customers']

        # Depot is node 0
        self.locations = [depot_location]

        # Generate customer locations
        for _ in range(num_customers):
            # Use sqrt for uniform distribution in circle
            r = radius_miles * np.sqrt(np.random.rand())
            theta = np.random.uniform(0, 2 * np.pi)

            # Calculate destination point
            origin = GeoPoint(depot_location)
            destination = distance(miles=r).destination(
                point=origin,
                bearing=np.degrees(theta)
            )

            customer_location = (destination.latitude, destination.longitude)
            self.locations.append(customer_location)

    def _compute_distance_matrix(self):
        """
        Compute pairwise geodesic distances (in miles) between all locations.
        Distance matrix is symmetric with zeros on diagonal.
        """
        n = len(self.locations)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = geodesic(self.locations[i], self.locations[j]).miles
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist  # Symmetric

    def _generate_customer_demands(self):
        """
        Generate demand for each customer.
        Currently uses uniform demand, but could be extended for variable demands.
        """
        num_customers = self.config['num_customers']
        demand_per_customer = self.config['demand_per_customer']

        # Depot has zero demand
        self.demands = [0] + [demand_per_customer] * num_customers

    def _generate_time_windows(self):
        """
        Generate random time windows for each customer.
        Each customer has an earliest and latest service start time.
        """
        num_customers = self.config['num_customers']
        day_start = self.config['day_start_hour']
        day_end = self.config['day_end_hour']
        max_window_length = self.config['max_time_window_hours']

        # Generate random earliest start times
        earliest_times = np.random.randint(
            low=day_start,
            high=day_end - 1,
            size=num_customers
        )

        # Generate random window durations
        window_durations = np.random.randint(
            low=1,
            high=max_window_length + 1,
            size=num_customers
        )

        # Calculate latest start times (capped at day end)
        latest_times = np.minimum(
            earliest_times + window_durations,
            day_end
        )

        # Depot has full day availability
        self.earliest_times = [day_start] + earliest_times.tolist()
        self.latest_times = [day_end] + latest_times.tolist()

    def _compute_travel_times(self):
        """
        Compute travel time matrix based on distances.
        Travel time is proportional to distance scaled by a factor.
        """
        travel_time_factor = self.config['travel_time_factor']
        radius_miles = self.config['radius_miles']

        # Travel time = factor * (distance / radius)
        self.travel_time_matrix = travel_time_factor * (
            self.distance_matrix / radius_miles
        )

    def get_num_nodes(self):
        """Returns total number of nodes (depot + customers)"""
        return len(self.locations)

    def get_num_customers(self):
        """Returns number of customers (excluding depot)"""
        return self.config['num_customers']

    def summary(self):
        """Print summary of generated problem instance"""
        print("=" * 60)
        print("CVRPTW Problem Instance Summary")
        print("=" * 60)
        print(f"Number of customers: {self.get_num_customers()}")
        print(f"Number of vehicles: {self.config['num_vehicles']}")
        print(f"Vehicle capacity: {self.config['vehicle_capacity']} units")
        print(f"Service duration: {self.config['service_duration_hours']} hours")
        print(f"Business hours: {self.config['day_start_hour']}:00 - {self.config['day_end_hour']}:00")
        print(f"Total demand: {sum(self.demands)} units")
        print(f"Total fleet capacity: {self.config['num_vehicles'] * self.config['vehicle_capacity']} units")
        print(f"Network radius: {self.config['radius_miles']} miles")
        print(f"Depot location: {self.config['depot_location']}")
        print("=" * 60)
