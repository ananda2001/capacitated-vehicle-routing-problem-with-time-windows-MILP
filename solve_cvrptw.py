"""
Main solver script for Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

This script demonstrates the complete workflow:
1. Configures problem parameters
2. Generates problem instance data
3. Builds and solves the MILP model
4. Visualizes and export results

Usage:
    python solve_cvrptw.py
"""

import sys
from config import get_config, validate_config, print_config
from data_generator import CVRPTWDataGenerator
from vrp_model import CVRPTWModel
from visualizer import RouteVisualizer


def main():
    """
    Main function to solve the CVRPTW problem.
    """
    print("\n" + "=" * 70)
    print(" " * 10 + "CAPACITATED VEHICLE ROUTING PROBLEM")
    print(" " * 15 + "WITH TIME WINDOWS (CVRPTW)")
    print("=" * 70)

    # ===== Step 1: Configuration =====
    print("\n[Step 1/5] Loading configuration...")

    # Get default configuration (or customize by passing overrides)
    # Example: config = get_config({'num_customers': 20, 'num_vehicles': 5})
    config = get_config()

    # Print configuration
    print_config(config)

    # Validate configuration
    is_valid, errors = validate_config(config)
    if not is_valid:
        print("Error!!! Configuration validation failed:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)

    print("Success!!! Configuration validated successfully")

    # ===== Step 2: Generate Problem Data =====
    print("\n[Step 2/5] Generating problem instance data...")

    data_generator = CVRPTWDataGenerator(config)
    data_generator.summary()

    print(f"Success!!! Generated {data_generator.get_num_customers()} customer locations")
    print(f"Success!!! Computed {data_generator.get_num_nodes()} x {data_generator.get_num_nodes()} distance matrix")
    print(f"Success!!! Generated time windows and demands")

    # ===== Step 3: Build MILP Model =====
    print("\n[Step 3/5] Building MILP model...")

    model = CVRPTWModel(data_generator)

    print(f"Success!!! Model built with {model.n} nodes")
    print(f"Success!!! Decision variables created: x (routing), s (time), y (load), slack (time window violations)")

    # Export model to .lp file for inspection
    model.export_model("cvrptw_model.lp")

    # ===== Step 4: Solve the Model =====
    print("\n[Step 4/5] Solving the optimization problem...")
    print("(This may take a few minutes depending on problem size...)")

    success = model.solve(time_limit=300, verbose=True)

    if not success:
        print("\nError!!! Failed to find a solution!")
        print("Try adjusting the configuration parameters.")
        sys.exit(1)

    # Print detailed solution
    model.print_solution()

    # ===== Step 5: Visualize Results =====
    print("\n[Step 5/5] Generating visualizations...")

    visualizer = RouteVisualizer(data_generator)

    # Plot the network (depot and customers)
    visualizer.plot_network(save_path="network.png")

    # Plot the optimal solution (routes)
    visualizer.plot_solution(model, save_path="solution.png")

    # ===== Summary =====
    import numpy as np
    total_slack = np.sum(model.solution['slack'])
    num_violations = np.sum(model.solution['slack'] > 0.01)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\nSuccess!!! Optimal solution found!")
    print(f"-> Total objective value: {model.solution['objective_value']:.2f}")
    print(f"-> Vehicles used: {len(model.routes)} out of {config['num_vehicles']} available")
    print(f"-> Time window violations: {total_slack:.2f} hours ({int(num_violations)} customers)")
    print(f"\nSuccess!!!Visualizations saved:")
    print(f" --> network.png")
    print(f" --> solution.png")
    print(f"\nSuccess!!! Model file saved:")
    print(f" --> cvrptw_model.lp")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nError!!! Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
