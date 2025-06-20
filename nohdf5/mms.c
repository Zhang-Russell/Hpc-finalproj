static char help[] = "Final, All-in-One Parallel 1D Heat Solver with Physics/MMS modes, Explicit/Implicit schemes, HDF5 Restart, and VTK output.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscviewerhdf5.h>

// Enum for run type: Physics simulation or Method of Manufactured Solutions
typedef enum {RUN_PHYSICS, RUN_MMS} RunType;
const char *const RunTypes[] = {"physics", "mms", "RunType", "RUN_", 0};

// Enum for time-stepping methods
typedef enum {TS_IMPLICIT, TS_EXPLICIT} TimeSteppingType;
const char *const TimeSteppingTypes[] = {"implicit", "explicit", "TimeSteppingType", "TS_", 0};

// Helper function to enforce boundary conditions on a vector
PetscErrorCode EnforceBoundaryConditions(DM da, Vec u)
{
    // Our chosen manufactured solution also has homogeneous (zero) boundary conditions,
    // so this function works for both physics and MMS modes.
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscScalar    *u_local; // MODIFIED FOR 1D: Was **u_local
    ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, u, &u_local);CHKERRQ(ierr);
    // MODIFIED FOR 1D: Single loop, check endpoints of the rod
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
        if (i == 0 || i == info.mx - 1) {
            u_local[i] = 0.0;
        }
    }
    ierr = DMDAVecRestoreArray(da, u, &u_local);CHKERRQ(ierr);
    return 0;
}

int main(int argc, char **args) {
    /* --- Variable Declarations --- */
    DM                  da;
    Vec                 u, u_old, f_source = NULL, u_exact = NULL, laplacian_u = NULL;
    Mat                 A = NULL, L = NULL;
    KSP                 ksp = NULL;
    PetscErrorCode      ierr;
    RunType             run_type = RUN_PHYSICS;
    TimeSteppingType    ts_type = TS_IMPLICIT;
    PetscInt            nx = 101, max_steps = 600, start_step = 0; // MODIFIED FOR 1D: ny removed
    PetscReal           final_time = 3, diffusivity = 0.01, dt = 0.0;
    PetscBool           flg_restart = PETSC_FALSE;
    char                checkpoint_file[] = "checkpoint.h5";
    PetscInt            checkpoint_interval = 20, vtk_interval = 20;

    /* --- Initialization and Option Parsing --- */
    ierr = PetscInitialize(&argc, &args, (char*)0, help); CHKERRQ(ierr);
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "1D Heat Equation Solver Options", "");
    ierr = PetscOptionsEnum("-run_type", "Running mode (physics or mms)", "main", RunTypes, (PetscEnum)run_type, (PetscEnum*)&run_type, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ts_type", "Time stepping method", "main", TimeSteppingTypes, (PetscEnum)ts_type, (PetscEnum*)&ts_type, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nx", "Grid points in x direction", "main", nx, &nx, NULL);CHKERRQ(ierr); // MODIFIED FOR 1D: ny option removed
    ierr = PetscOptionsReal("-final_time", "Final simulation time", "main", final_time, &final_time, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-max_steps", "Number of time steps (used if -dt is not set)", "main", max_steps, &max_steps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt", "Time step size (overrides max_steps)", "main", dt, &dt, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-restart", "Enable restarting from checkpoint file", "main", flg_restart, &flg_restart, NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    /* --- DMDA and Vector Setup --- */
    // MODIFIED FOR 1D: Using DMDACreate1d
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, nx, 1, 1, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    // MODIFIED FOR 1D: Setting 1D coordinates
    ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u, "Numerical_Solution"); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &u_old); CHKERRQ(ierr);
    if (run_type == RUN_MMS) {
        ierr = VecDuplicate(u, &f_source); CHKERRQ(ierr);
        ierr = VecDuplicate(u, &u_exact); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)u_exact, "Exact_Solution"); CHKERRQ(ierr);
    }

    /* --- Initial State --- */
    if (flg_restart) {
        PetscViewer hdf5_viewer;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Attempting to restart from %s\n", checkpoint_file); CHKERRQ(ierr);
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, checkpoint_file, FILE_MODE_READ, &hdf5_viewer); CHKERRQ(ierr);
        ierr = VecLoad(u, hdf5_viewer); CHKERRQ(ierr);
        Vec v_step;
        ierr = VecCreate(PETSC_COMM_WORLD, &v_step); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)v_step, "step"); CHKERRQ(ierr);
        ierr = VecLoad(v_step, hdf5_viewer); CHKERRQ(ierr);
        PetscMPIInt rank;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
        if (rank == 0) {
            const PetscScalar *step_array;
            ierr = VecGetArrayRead(v_step, &step_array); CHKERRQ(ierr);
            start_step = (PetscInt)PetscRealPart(step_array[0]);
            ierr = VecRestoreArrayRead(v_step, &step_array); CHKERRQ(ierr);
        }
        ierr = MPI_Bcast(&start_step, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = VecDestroy(&v_step); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&hdf5_viewer); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Successfully restarted from step %d\n", start_step); CHKERRQ(ierr);
    } else {
        DMDALocalInfo info;
        PetscScalar   *u_local; // MODIFIED FOR 1D: was **u_local
        PetscReal     dx = 1.0/(nx-1);
        ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, u, &u_local); CHKERRQ(ierr);
        // MODIFIED FOR 1D: Single loop for initial condition
        for (PetscInt i=info.xs; i<info.xs+info.xm; i++) {
            PetscReal x=i*dx;
            if (run_type == RUN_PHYSICS) {
                // A 'hot spot' in the middle of the rod
                if (x>=0.4&&x<=0.6) u_local[i] = 100.0; else u_local[i] = 0.0;
            } else { // RUN_MMS
                // 1D manufactured solution initial condition
                u_local[i] = sin(PETSC_PI * x);
            }
        }
        ierr = DMDAVecRestoreArray(da, u, &u_local); CHKERRQ(ierr);
    }
    ierr = EnforceBoundaryConditions(da, u); CHKERRQ(ierr);

    /* --- Pre-Loop Setup --- */
    if (dt > 0.0) { max_steps = (PetscInt)(final_time / dt); } else { dt = final_time / max_steps; }
    
    if (ts_type == TS_IMPLICIT) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting up for IMPLICIT method, dt = %g, steps = %d\n", (double)dt, max_steps);CHKERRQ(ierr);
        ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
        DMDALocalInfo info; ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        // MODIFIED FOR 1D: 3-point stencil
        MatStencil row, col[3]; 
        PetscScalar v[3];
        PetscReal dx = 1.0/(nx-1), alpha = diffusivity*dt/(dx*dx);
        // MODIFIED FOR 1D: Single loop for matrix assembly
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            row.i = i;
            if (i==0||i==nx-1) { // Boundary condition
                v[0] = 1.0; 
                ierr = MatSetValuesStencil(A, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr); 
            } else { // Interior point
                v[0]=1.0+2.0*alpha; col[0]=row; 
                v[1]=-alpha;        col[1].i=i-1;
                v[2]=-alpha;        col[2].i=i+1;
                ierr = MatSetValuesStencil(A, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    } else { // Explicit
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting up for EXPLICIT method, dt = %g, steps = %d\n", (double)dt, max_steps);CHKERRQ(ierr);
        PetscReal dx = 1.0/(nx-1);
        // MODIFIED FOR 1D: Stability condition
        PetscReal stability_factor = diffusivity * dt / (dx*dx);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Stability Factor: %g (must be <= 0.5)\n", (double)stability_factor); CHKERRQ(ierr);
        if (stability_factor > 0.5) { ierr = PetscPrintf(PETSC_COMM_WORLD, "WARNING: Stability condition not met!\n");CHKERRQ(ierr); }
        ierr = DMCreateMatrix(da, &L); CHKERRQ(ierr);
        ierr = VecDuplicate(u, &laplacian_u); CHKERRQ(ierr);
        DMDALocalInfo info; ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        // MODIFIED FOR 1D: 3-point stencil for Laplacian
        MatStencil row, col[3]; 
        PetscScalar v[3];
        // MODIFIED FOR 1D: Single loop for matrix assembly
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            row.i = i;
            if (i==0||i==nx-1) { 
                v[0] = 0.0; ierr = MatSetValuesStencil(L, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
            } else {
                v[0]=-2.0/(dx*dx); col[0]=row; 
                v[1]=1.0/(dx*dx);  col[1].i=i-1; 
                v[2]=1.0/(dx*dx);  col[2].i=i+1;
                ierr = MatSetValuesStencil(L, 1, &row, 3, col, v, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
        ierr = MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    
    /* --- Main Time-Stepping Loop --- */
    for (PetscInt step = start_step; step < max_steps; ++step) {
        PetscReal current_time = (step + 1) * dt;

        if (run_type == RUN_MMS) {
            DMDALocalInfo info; ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
            PetscScalar *f_local; ierr = DMDAVecGetArray(da, f_source, &f_local); CHKERRQ(ierr); // MODIFIED: 1D array
            PetscReal dx = 1.0/(nx-1);
            // MODIFIED FOR 1D: Single loop, 1D MMS source term
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                PetscReal x = i * dx;
                f_local[i] = (diffusivity*PETSC_PI*PETSC_PI - 1.0) * sin(PETSC_PI*x) * exp(-current_time);
            }
            ierr = DMDAVecRestoreArray(da, f_source, &f_local); CHKERRQ(ierr);
        }

        if (ts_type == TS_IMPLICIT) {
            ierr = VecCopy(u, u_old); CHKERRQ(ierr);
            if (run_type == RUN_MMS) { ierr = VecAXPY(u_old, dt, f_source); CHKERRQ(ierr); } // Add source to RHS
            ierr = KSPSolve(ksp, u_old, u); CHKERRQ(ierr);
        } else { // Explicit
            ierr = MatMult(L, u, laplacian_u); CHKERRQ(ierr);
            ierr = VecAXPY(u, dt * diffusivity, laplacian_u); CHKERRQ(ierr); // Diffusion part
            if (run_type == RUN_MMS) { ierr = VecAXPY(u, dt, f_source); CHKERRQ(ierr); } // Source part
            ierr = EnforceBoundaryConditions(da, u);CHKERRQ(ierr);
        }

        // --- CHECKPOINTING SECTION DISABLED ---
        /*
        if ((step + 1) % checkpoint_interval == 0 && (step + 1) < max_steps) {
            PetscViewer hdf5_viewer;
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Writing checkpoint at step %ld to %s\n", (long)(step + 1), checkpoint_file); CHKERRQ(ierr);
            ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, checkpoint_file, FILE_MODE_WRITE, &hdf5_viewer); CHKERRQ(ierr);
            ierr = VecView(u, hdf5_viewer); CHKERRQ(ierr);
            Vec v_step; PetscScalar *step_array;
            ierr = VecCreate(PETSC_COMM_WORLD, &v_step); CHKERRQ(ierr);
            ierr = VecSetSizes(v_step, PETSC_DECIDE, 1); CHKERRQ(ierr);
            ierr = VecSetUp(v_step); CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject)v_step, "step"); CHKERRQ(ierr);
            PetscMPIInt rank; ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
            if (rank == 0) {
                ierr = VecGetArray(v_step, &step_array); CHKERRQ(ierr);
                step_array[0] = step + 1;
                ierr = VecRestoreArray(v_step, &step_array); CHKERRQ(ierr);
            }
            ierr = VecView(v_step, hdf5_viewer); CHKERRQ(ierr);
            ierr = VecDestroy(&v_step); CHKERRQ(ierr);
            ierr = PetscViewerDestroy(&hdf5_viewer); CHKERRQ(ierr);
        }
        */

        if (vtk_interval > 0 && ((step + 1) % vtk_interval == 0 || (step + 1) == max_steps)) {
             PetscViewer vtk_viewer; char filename[PETSC_MAX_PATH_LEN];
             ierr = PetscSNPrintf(filename, sizeof(filename), "solution-%04ld.vts", (long)(step + 1)); CHKERRQ(ierr);
             ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &vtk_viewer); CHKERRQ(ierr);
             ierr = VecView(u, vtk_viewer); CHKERRQ(ierr);
             ierr = PetscViewerDestroy(&vtk_viewer); CHKERRQ(ierr);
        }
    }
    
    /* --- Post-Loop: Error Calculation for MMS --- */
    if (run_type == RUN_MMS) {
        DMDALocalInfo info;
        PetscScalar   *uexact_local; // MODIFIED FOR 1D: was **uexact_local
        PetscReal     dx = 1.0/(nx-1);
        ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, u_exact, &uexact_local); CHKERRQ(ierr);
        // MODIFIED FOR 1D: Single loop, 1D exact solution
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            PetscReal x = i * dx;
            uexact_local[i] = sin(PETSC_PI * x) * exp(-final_time);
        }
        ierr = DMDAVecRestoreArray(da, u_exact, &uexact_local); CHKERRQ(ierr);
        
        PetscReal error_norm;
        ierr = VecAXPY(u, -1.0, u_exact); // u = u - u_exact (error vector)
        ierr = VecNorm(u, NORM_INFINITY, &error_norm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------\n"); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MMS Verification Result:\n"); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "L-infinity Error at T=%.2f with nx=%d, dt=%.6f is: %g\n", (double)final_time, nx, (double)dt, (double)error_norm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------\n"); CHKERRQ(ierr);
    }
    
    /* --- Final Output and Cleanup --- */
    char filename[PETSC_MAX_PATH_LEN];
    PetscViewer vtk_viewer;
    if (ts_type == TS_IMPLICIT) { ierr = PetscSNPrintf(filename, sizeof(filename), "implicit_solution.vts"); CHKERRQ(ierr); }
    else { ierr = PetscSNPrintf(filename, sizeof(filename), "explicit_solution.vts"); CHKERRQ(ierr); }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Writing final solution to %s\n", filename); CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &vtk_viewer); CHKERRQ(ierr);
    ierr = VecView(u, vtk_viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtk_viewer); CHKERRQ(ierr);

    if(A) {ierr = MatDestroy(&A); CHKERRQ(ierr); ierr = KSPDestroy(&ksp); CHKERRQ(ierr);}
    if(L) {ierr = MatDestroy(&L); CHKERRQ(ierr); ierr = VecDestroy(&laplacian_u); CHKERRQ(ierr);}
    ierr = VecDestroy(&u); CHKERRQ(ierr); ierr = VecDestroy(&u_old); CHKERRQ(ierr);
    if(f_source) {ierr = VecDestroy(&f_source); CHKERRQ(ierr);}
    if(u_exact) {ierr = VecDestroy(&u_exact); CHKERRQ(ierr);}
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
