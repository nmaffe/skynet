import os
import numpy as np
import matplotlib.pyplot as plt

"""
The purpose of this program is to fetch the glacier metadata 
at random locations and produce ice thickness predictions at these locations 
using the the GNN.
Input: glacier geometry, GNN trained model
Output: some kind of array of ice thickness predictions that can be used by the inpainting model.  
"""
