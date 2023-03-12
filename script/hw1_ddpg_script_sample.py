import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings('ignore')

from assetAllocationDDPG import assetAllocationDDPG

'''
This is a sample script to demonstrate how to use assetAllocationDDPG class to solve the problem in MAFS5370
assignment 1.
'''


if __name__ == "__main":

    # Specify the parameters
    initial_wealth = 1
    p = 0.5
    a = 1.2
    b = 0.8
    risk_free = 1
    T = 10

    # Initialize the problem
    asset_allocation = assetAllocationDDPG(initial_wealth, p, a, b, risk_free, T)

    # Run asset allocation to solve the problem
    asset_allocation.asset_allocation_ddpg()

    # Output action and Q value
    # Using default window size which is 100 now
    action, q_value = asset_allocation.output()





