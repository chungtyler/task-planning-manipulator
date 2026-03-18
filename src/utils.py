import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

class StateLogger:
    """
    Log and display state data results from runtime
    """
    def __init__(self, state_types=['time', 'position', 'velocity', 'force'], units=['s', 'm', 'm/s', 'N']):
        self.state_types = state_types
        self.units = units
        self.state_history = {state: [] for state in self.state_types}
        self.target_history = {state: [] for state in self.state_types}

    # Log state data
    def log(self, state_data, target_data):
        for state in self.state_types:
            self.state_history[state].append(state_data[state])
            self.target_history[state].append(target_data[state])

    # Plot state data
    def plot(self):
        # Convert lists to arrays
        state_array = []
        target_array = []
        for state in self.state_types:
            state_data = np.vstack(self.state_history[state])
            state_array.append(state_data)

            target_data = np.vstack(self.target_history[state])
            target_array.append(target_data)
        
        # Plot each state type in cartesian coordinates
        labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'royalblue']
        plt.rcParams['lines.linewidth'] = 2
        for index, state in enumerate(self.state_types[1:]):
            _, ax = plt.subplots()
            for i in range(3):
                ax.plot(state_array[0], state_array[index+1][:, i], label=labels[i], color=colors[i])
                if target_array[index+1].shape[1] != 0:
                  ax.plot(target_array[0], target_array[index+1][:, i], label=f"{labels[i]} Desired", color=colors[i], linestyle='--', alpha=0.5)
            
            ax.set_ylabel(f"{state.capitalize()} [{self.units[index+1]}]")
            ax.set_xlabel(f"{self.state_types[0].capitalize()} [{self.units[0]}]")
            ax.set_xlim(0, state_array[0][-1]+0.01)

            plt.grid(True)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncols=3)
            plt.tight_layout()
        plt.show()

# Generate marker
def create_marker(viewer, mujoco):
    scene = viewer.user_scn
    num_geoms = scene.ngeom
    geom = scene.geoms[num_geoms]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.04, 0.0, 0.0]),  # size
        np.array([0, 0, 0], dtype=np.float64),  # pos
        np.eye(3).flatten(),  # rotation matrix (identity)
        np.array([1, 0, 0.9, 0.5], dtype=np.float32)  # rgba
    )
    scene.ngeom += 1
    return geom
    
# Grab yaml content as python list
def get_yaml_content(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = yaml.safe_load(f)
            return content