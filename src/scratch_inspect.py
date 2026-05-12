import sys
import importlib.util
sys.path.append('.')

# Fake pkg_resources for Py_FS
import types
if 'pkg_resources' not in sys.modules:
    sys.modules['pkg_resources'] = types.ModuleType('pkg_resources')

import builtins
builtins.input = lambda prompt="": "0.7"

import sklearn.model_selection
original_split = sklearn.model_selection.train_test_split
def safe_split(*args, **kwargs):
    if 'stratify' in kwargs: del kwargs['stratify']
    kwargs['test_size'] = 0.3
    return original_split(*args, **kwargs)
sklearn.model_selection.train_test_split = safe_split

from Py_FS.wrapper.nature_inspired import PSO
import numpy as np

# Create dummy data to test PSO quickly
X_train = np.random.rand(20, 5)
y_train = np.random.randint(0, 2, 20)

sol_pso = PSO(num_agents=5, max_iter=2, train_data=X_train, train_label=y_train, save_conv_graph=False)
print("ATTRIBUTES:")
print(dir(sol_pso))
