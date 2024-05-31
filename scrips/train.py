## Note might not work

import numpy as np
import matplotlib.pyplot as pyplot
from ..src.main.py import *
from ..src.utils.py import *  ## TODO: Fix

mdp = Admission_MDP(goal=goal, admission_decay=admission_decay, acceptance_probabilities=acceptance_probabilities, new_students=new_students, q_table = {})
q_values = mdp.run(epochs=200)  ##TODO: Fix