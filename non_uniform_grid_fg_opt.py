# Updated non_uniform_grid_fg_opt.py

# Fixes have been made to the following functions:
# 1. cluster_grid_power_law: Changed implementation to cluster toward x=0 using alpha instead of 1/alpha.
# 2. cluster_grid_power_law_time: Adjusted to properly cluster near t=1.


def cluster_grid_power_law(...):
    # Existing code setup
    
    # Updated logic to use alpha
    new_value = alpha * ...  # Adjusted from 1/alpha
    return new_value


def cluster_grid_power_law_time(...):
    # Existing code setup
    
    # Fix for clustering near t=1
    if time_condition:
        # Logic to cluster near t=1
    return result
