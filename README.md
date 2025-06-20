# MAE 5032 Final Project: A Parallel 1D Transient Heat Equation Solver

This project presents a parallel numerical solver for the one-dimensional transient heat equation on a unit interval domain Omega := (0, 1). The solver simulates the temperature evolution along a 1D rod. This software was developed to fulfill the requirements of the MAE 5032 High-Performance Computing for Applications final project.

The program is written in C and utilizes the PETSc library to handle parallel data structures, matrix assembly, and the solution of linear systems. It is designed for execution on high-performance computing clusters using MPI.

When testing locally, I encountered an error when nx is too large (5000000), which is due to too much data written to the checkpoint file, so please use the code in the nohdf5 folder.
In the Taiyi test, the Makefile used is linked to the petsc library of the teaching assistant, so please use the Makefile in the ty folder
If you want to change the grid(nx),you can change it in run.lsf
## Core Features

**1D Heat Equation Solver**: Implements a parallel solver for the transient heat equation, based on the Finite Difference Method for spatial discretization.
**Dual Time-Stepping Schemes**: Includes both **Explicit Euler** and **Implicit Euler** methods for time integration, which can be selected at runtime.
**Method of Manufactured Solutions (MMS)**: Features a dedicated verification mode (`-run_type mms`) to test the code's accuracy against a known analytical solution and automatically calculate the L infinity error.
**HDF5 Restart Capability**: Supports checkpointing and restarting simulations via HDF5, allowing long computations to be resumed if interrupted.
**Parallel by Design**: Built from the ground up for parallel execution using PETSc's DMDA objects for 1D domain decomposition.
**Advanced Visualization**: Outputs results in VTK format, suitable for detailed analysis and visualization in professional tools like **ParaView**.
**Extensive Configurability**: All key parameters, including grid size, time step, solver type, and run mode, are configurable via command-line arguments.

## Dependencies

* A working MPI implementation (e.g., MPICH, OpenMPI).
* The PETSc library, configured with HDF5 support.
* The `PETSC_DIR` and `PETSC_ARCH` environment variables must be correctly set, or the provided `Makefile` must be pointed to the correct installation path.

## How to Build

The project includes a `Makefile` for easy, robust compilation.

1.  **Set Environment**: Ensure your environment is configured to use the correct PETSc installation. On the Tai-Yi cluster, this involves loading the appropriate modules and potentially setting `PETSC_DIR`.
2.  **Compile**: From the project's root directory, simply run `make mms`.
    ```bash
    make
    ```
    This will generate the executable file (e.g., `mms`).
3.  **Clean**: To remove the executable and object files, run:
    ```bash
    make clean
    ```

## How to Run & Perform Required Tests

The program is run via `mpiexec`. Below are examples for each required test.

### 1. Physics Simulation (for Visualization)

This mode simulates the diffusion of a central hot spot on the rod.
```bash
# Run with implicit method on 4 cores
mpiexec -n 4 ./mms -run_type physics -ts_type implicit -nx 201
```

### 2. Numerical Stability Test 

Fix the spatial grid and vary the time step `dt` to find the stability limit of the explicit method.

```bash
# Fix grid to nx=101. The theoretical stability limit is dt <= 0.005.

# These should be STABLE
mpiexec -n 2 ./mms -run_type physics -ts_type explicit -nx 101 -dt 0.0049
mpiexec -n 2 ./mms -run_type physics -ts_type explicit -nx 101 -dt 0.005

# This should DIVERGE
mpiexec -n 2 ./mms -run_type physics -ts_type explicit -nx 101 -dt 0.0051
```

### 3. Code Verification (MMS) Test 

Run in `mms` mode to get the L-infinity error for convergence rate analysis.

**To find Spatial Order alpha**: Fix `dt` to be very small and vary `nx`.
```bash
mpiexec -n 4 ./mms -run_type mms -ts_type implicit -nx 41 -dt 0.0001
mpiexec -n 4 ./mms -run_type mms -ts_type implicit -nx 81 -dt 0.0001
mpiexec -n 4 ./mms -run_type mms -ts_type implicit -nx 161 -dt 0.0001
```
Record the L-infinity error from the terminal output for each run.

**To find Temporal Order beta**: Fix `nx` to be very large and vary `dt`.
```bash
mpiexec -n 4 ./mms -run_type mms -ts_type implicit -nx 201 -dt 0.01
mpiexec -n 4 ./mms -run_type mms -ts_type implicit -nx 201 -dt 0.005
mpiexec -n 4 ./mms -run_type mms -ts_type implicit -nx 202 -dt 0.0025
```
Record the L-infinity error for each run.

### 4. Parallel Performance Test (on Cluster) 

Use the provided LSF job script (`run_job.lsf`) to run these tests on the cluster.

**Strong Scaling**: Fix the problem size and increase the number of cores.
```bash
# Example command inside the job script
mpiexec -n $SLURM_NTASKS ./mms -run_type physics -ts_type implicit -nx 4097 -log_view
```
Submit jobs with different core counts (e.g., 2, 4, 8, 16...) and analyze the timing data from the `-log_view` output.

**Weak Scaling**: Keep the problem size per core constant.
```bash
# 2 cores, N/p = 2048/core
mpiexec -n 2 ./mms -run_type physics -ts_type implicit -nx 4097 -log_view

# 4 cores, N/p = 2048/core
mpiexec -n 4 ./mms -run_type physics -ts_type implicit -nx 8193 -log_view
```
Analyze the timing data from the `-log_view` output.

### 5. HDF5 Restart Test 
```bash
# 1. Run for 15 steps (this will save a checkpoint at step 10)
mpiexec -n 2 ./mms -max_steps 15

# 2. Resume from the checkpoint and run until step 30
mpiexec -n 2 ./mms -max_steps 30 -restart
```

## Visualization

The program outputs `.vts` files that can be opened in ParaView. For a 1D problem, the output will be a line.
1.  Open ParaView and load the `.vts` file.
2.  Click **Apply**.
3.  In the Display properties, you may need to increase the "Line Width" to see the line clearly.
4.  In the Coloring section, select **Temperature** to color the line according to the solution values.
