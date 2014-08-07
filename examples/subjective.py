

import os
import sys
# local devel testing without installing the package
sys.path = [os.path.abspath(os.path.join(os.getcwd(),'..')), ] + sys.path


import navmet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load some sample agents
agents = np.loadtxt('sample_agents.txt')

# plot their configuration
plt.plot(agents[:, 0], agents[:, 1], ls='', marker='8')
plt.show()
