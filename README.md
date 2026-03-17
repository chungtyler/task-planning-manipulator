# Z1-Impedance-Control
Cartesian Impedance Control on Z1 Unitree Manipulator

https://youtu.be/z6i7P8Pc98Q

https://github.com/user-attachments/assets/baeaf634-9c52-4e88-8485-fbb11f3123eb

# Installation
Clone the repository, then navigate to the main directory
```bash
git clone https://github.com/chungtyler/z1-impedance-control.git
cd z1-impedance-control
```

Setup conda environment
```bash
conda env create -f environment.yaml
conda activate z1
```

# Run Simulation
To run the simulator, enter:
```bash
python -m scripts.run_simulation
```

To edit controller stiffness, change the constant multiplier for `K_d_translation` and `K_d_rotation`.  
The desired target states can also be defined under `pose_d` and `twist_d`.  
Set the simulation time limit under the `time_limit` variable.  
To add potential fields, edit the `config.potential_field.yaml` (Uncomment potential field code in the setup and control_loop()).  
To edit the scene, edit `env.scene.xml`.

# Run Hardware
To run in hardware on the physical Unitree Z1, enter:
```bash
python -m scripts.run_hardware
```

Ensure that the controller is evaluated in the simulator before physical hardware testing
Edit the same parameters in [Run Simulation](#run-simulation)


