# Capacitated Vehicle Routing Problem With Time Windows As A Mixed Integer Linear Program

A MILP formulation of the capacitated vehicle routing problem with time windows (CVRPTW). I use soft time windows with a weighted penalty in the objective function to discourage lateness while maintaining feasibility at all times. I used [this codebase](https://github.com/sudhan-bhattarai/vehicle_routing_problem?tab=readme-ov-file) as a starting point for formulating and implementing my problem. This was a class project I did for **ECE 5500: Nonlinear and Dynamic Programming** at Ohio State University.

To run the code, first please make sure you have the following Python libraries installed: ``gurobipy, geopandas, geopy contextily, Shapely, matplotlib, numpy``, or just run the command ``pip install -r requirements.txt`` in your terminal window. You will need a valid Gurobi license to use Gurobi after installing ``gurobipy``, for more information please click on [this link](https://www.gurobi.com/solutions/licensing/).

After making sure all the dependencies are installed, all you have to do is run ``python solve_cvrptw.py`` in your terminal after selecting the parameters of your choice (number of vehicles, number of customers, etc). For parameter selection, there are two options here: 

1) The easiest way is to modify the default configuration setup (marked as ``DEFAULT_CONFIG``) in ``config.py``, following which you can run ``python solve_cvrptw.py``.

2) You can also make your parameter selection in ``solve_cvrptw.py`` itself, in line 35 (for example, ``config = get_config({'num_customers': 20, 'num_vehicles': 5})`` for 20 customers and 5 vehicles, with the remaining parameters being the same as in ``DEFAULT_CONFIG`` in ``config.py``).

After running the code, the model will the saved in the project directory as ``cvrptw_model.lp``, along with the plots of the customer nodes (locations) and the optimal solution (as ``network.png`` and ``solution.png`` respectively). Sample plots of a test case (whose parameters can be found by visiting ``DEFAULT_CONFIG`` in ``config.py``) have been included in the **Test** folder in this repository, or by clicking on [this link](https://github.com/ananda2001/capacitated-vehicle-routing-problem-with-time-windows-MILP/tree/main/Test).

For more details about the problem including the mathematical formulation and test results, please read the reported documentation in the ``Documentation.py`` file included in this repository or by clicking on [this link](https://github.com/ananda2001/capacitated-vehicle-routing-problem-with-time-windows-MILP/blob/main/Documentation.pdf).
