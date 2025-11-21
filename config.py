"""
Configuration file for CVRPTW problem parameters.

All problem parameters are defined here for easy modification.
"""

# Default configuration for CVRPTW problem
DEFAULT_CONFIG = {
    # --- Problem Size ---
    'num_vehicles': 5,              # Number of vehicles in the fleet
    'num_customers': 80,             # Number of customers to serve
    'vehicle_capacity': 16,           # Maximum capacity per vehicle (units)
    'demand_per_customer': 1,        # Demand at each customer (units)

    # --- Geographic Parameters ---
    # (Amazon Fulfillment Center CMH1, Etna, Ohio)
    'depot_location': (39.9543, -82.6860),  # (lat, lon)
    'radius_miles': 80,              # Network radius around depot (miles) - regional delivery area

    # --- Time Parameters ---
    'day_start_hour': 8,             # Business day start time (e.g., 8 = 8:00 AM)
    'day_end_hour': 18,              # Business day end time (e.g., 18 = 6:00 PM)
    'service_duration_hours': 0.25,  # Fixed service time at each customer (hours)
    'max_time_window_hours': 4,      # Maximum time window length (hours)
    'travel_time_factor': 0.6,       # Factor relating distance to travel time (~25 mph avg speed)

    # --- Time Window Penalty ---
    'time_window_penalty': 1000.0,   # Penalty weight for late deliveries (soft constraint)
                                     # Higher value = stronger preference for on-time delivery
                                     # Lower value = more willing to trade lateness for shorter routes

    # --- Other Parameters ---
    'random_seed': 42,               # Random seed for reproducibility
}


def get_config(overrides=None):
    """
    Gets configuration dictionary with optional parameter overrides.

    Args:
        overrides (dict): Dictionary of parameter overrides

    Returns:
        dict: Complete configuration dictionary

    Example:
        >>> config = get_config({'num_customers': 20, 'num_vehicles': 5})
    """
    config = DEFAULT_CONFIG.copy()

    if overrides:
        config.update(overrides)

    return config


def validate_config(config):
    """
    Validates that the configuration parameters are feasible.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []

    # Check fleet capacity vs total demand
    total_capacity = config['num_vehicles'] * config['vehicle_capacity']
    total_demand = config['num_customers'] * config['demand_per_customer']

    if total_capacity < total_demand:
        errors.append(
            f"Insufficient fleet capacity! "
            f"Total capacity ({total_capacity}) < Total demand ({total_demand}). "
            f"Increase num_vehicles or vehicle_capacity."
        )

    # Check time parameters
    if config['day_start_hour'] >= config['day_end_hour']:
        errors.append(
            f"Invalid business hours: day_start_hour ({config['day_start_hour']}) "
            f"must be less than day_end_hour ({config['day_end_hour']})"
        )

    if config['max_time_window_hours'] > (config['day_end_hour'] - config['day_start_hour']):
        errors.append(
            f"max_time_window_hours ({config['max_time_window_hours']}) cannot exceed "
            f"business day length ({config['day_end_hour'] - config['day_start_hour']} hours)"
        )

    # Check positive values
    positive_params = [
        'num_vehicles', 'num_customers', 'vehicle_capacity',
        'demand_per_customer', 'radius_miles', 'service_duration_hours'
    ]

    for param in positive_params:
        if config[param] <= 0:
            errors.append(f"{param} must be positive, got {config[param]}")

    is_valid = len(errors) == 0
    return is_valid, errors


def print_config(config):
    """
    Pretty print the configuration parameters.

    Args:
        config (dict): Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("CVRPTW CONFIGURATION")
    print("=" * 60)
    print("\nProblem Size:")
    print(f"  Number of vehicles:        {config['num_vehicles']}")
    print(f"  Number of customers:       {config['num_customers']}")
    print(f"  Vehicle capacity:          {config['vehicle_capacity']} units")
    print(f"  Demand per customer:       {config['demand_per_customer']} units")

    print("\nGeographic Parameters:")
    print(f"  Depot location:            {config['depot_location']}")
    print(f"                             (Amazon FC CMH1, Etna, OH)")
    print(f"  Network radius:            {config['radius_miles']} miles")

    print("\nTime Parameters:")
    print(f"  Business hours:            {config['day_start_hour']}:00 - {config['day_end_hour']}:00")
    print(f"  Service duration:          {config['service_duration_hours']} hours")
    print(f"  Max time window length:    {config['max_time_window_hours']} hours")
    print(f"  Travel time factor:        {config['travel_time_factor']}")
    print(f"  Time window penalty:       {config['time_window_penalty']} (soft constraint)")

    print("\nOther:")
    print(f"  Random seed:               {config['random_seed']}")

    # Validation check
    total_capacity = config['num_vehicles'] * config['vehicle_capacity']
    total_demand = config['num_customers'] * config['demand_per_customer']

    print("\nFeasibility Check:")
    print(f"  Total fleet capacity:      {total_capacity} units")
    print(f"  Total demand:              {total_demand} units")
    print(f"  Capacity margin:           {total_capacity - total_demand} units")

    if total_capacity >= total_demand:
        print("  Status:                    ✓ Feasible")
    else:
        print("  Status:                    ✗ INFEASIBLE - Insufficient capacity!")

    print("=" * 60 + "\n")
