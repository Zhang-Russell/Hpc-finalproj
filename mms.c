static char help[] = "Final, complete 1D Parallel Heat Solver with All Features: Physics/MMS, Explicit/Implicit, HDF5 Restart, and VTK output.\n\n";

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

// Helper function to enforce boundary conditions for the 1D rod
PetscErrorCode EnforceBoundaryConditions(DM da, Vec u)
{
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscScalar    *u_local;
    ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, u, &u_local);CHKERRQ(ierr);
    // Set BCs at the two ends of the rod
    if (info.xs == 0) u_local[0] = 0.0; // Left end (x=0)
    if (info.xs + info.xm == info.mx) u_local[info.xm - 1] = 0.0; // Right end (x=1)
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
    PetscInt            nx = 101, max_steps = 200, start_step = 0;
    PetscReal           final_time = 0.2, diffusivity = 0.01, dt = 0.0;
    PetscBool           flg_restart = PETSC_FALSE;
    char                checkpoint_file[] = "checkpoint.h5";
    PetscInt            checkpoint_interval = 20, vtk_interval = 20;

    /* --- Initialization and Option Parsing --- */
    ierr = PetscInitialize(&argc, &args, (char*)0, help); CHKERRQ(ierr);
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "1D Heat Equation Solver Options", "");
    ierr = PetscOptionsEnum("-run_type", "Running mode (physics or mms)", "main", RunTypes, (PetscEnum)run_type, (PetscEnum*)&run_type, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ts_type", "Time stepping method", "main", TimeSteppingTypes, (PetscEnum)ts_type, (PetscEnum*)&ts_type, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nx", "Grid points in x direction", "main", nx, &nx, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-final_time", "Final simulation time", "main", final_time, &final_time, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-max_steps", "Number of time steps (used if -dt is not set)", "main", max_steps, &max_steps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt", "Time step size (overrides max_steps)", "main", dt, &dt, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-restart", "Enable restarting from checkpoint file", "main", flg_restart, &flg_restart, NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    /* --- DMDA and Vector Setup --- */
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, nx, 1, 1, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u, "Temperature"); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &u_old); CHKERRQ(ierr);
    if (run_type == RUN_MMS) {
        ierr = VecDuplicate(u, &f_source); CHKERRQ(ierr);
        ierr = VecDuplicate(u, &u_exact); CHKERRQ(ierr);
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
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Successfully restarted from step %ld\n", (long)start_step); CHKERRQ(ierr);
    } else {
        DMDALocalInfo info;
        PetscScalar   *u_local;
        PetscReal     dx = 1.0/(nx-1);
        ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, u, &u_local); CHKERRQ(ierr);
        for (PetscInt i=info.xs; i<info.xs+info.xm; i++) {
            PetscReal x = i*dx;
            if (run_type == RUN_PHYSICS) {
                if (x>=0.4 && x<=0.6) u_local[i] = 100.0; else u_local[i] = 0.0;
            } else { // RUN_MMS
                u_local[i] = sin(PETSC_PI * x);
            }
        }
        ierr = DMDAVecRestoreArray(da, u, &u_local); CHKERRQ(ierr);
    }
    ierr = EnforceBoundaryConditions(da, u); CHKERRQ(ierr);

    /* --- Pre-Loop Setup --- */
    if (dt > 0.0) { max_steps = (PetscInt)(final_time / dt); } else { dt = final_time / max_steps; }
    PetscReal dx = 1.0 / (nx - 1);
    
    if (ts_type == TS_IMPLICIT) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting up for 1D IMPLICIT method...\n");CHKERRQ(ierr);
        ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
        PetscReal alpha = diffusivity * dt / (dx * dx);
        for (PetscInt i = 0; i < nx; i++) {
            if (i == 0 || i == nx - 1) {
                ierr = MatSetValue(A, i, i, 1.0, INSERT_VALUES); CHKERRQ(ierr);
            } else {
                PetscScalar vals[3] = {-alpha, 1.0 + 2.0 * alpha, -alpha};
                PetscInt    cols[3] = {i - 1, i, i + 1};
                ierr = MatSetValues(A, 1, &i, 3, cols, vals, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    } else { // Explicit
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting up for 1D EXPLICIT method...\n");CHKERRQ(ierr);
        PetscReal stability_factor = diffusivity * dt / (dx * dx);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Stability Factor: %g (must be <= 0.5 for 1D)\n", (double)stability_factor); CHKERRQ(ierr);
        if (stability_factor > 0.5) { ierr = PetscPrintf(PETSC_COMM_WORLD, "WARNING: Stability condition not met!\n");CHKERRQ(ierr); }
        ierr = DMCreateMatrix(da, &L); CHKERRQ(ierr);
        ierr = VecDuplicate(u, &laplacian_u); CHKERRQ(ierr);
        for (PetscInt i = 0; i < nx; i++) {
            if (i > 0 && i < nx - 1) {
                PetscScalar vals[3] = {1.0/(dx*dx), -2.0/(dx*dx), 1.0/(dx*dx)};
                PetscInt    cols[3] = {i - 1, i, i + 1};
                ierr = MatSetValues(L, 1, &i, 3, cols, vals, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
        ierr = MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    
    /* --- Main Time-Stepping Loop --- */
    for (PetscInt step = start_step; step < max_steps; ++step) {
        PetscReal time_n = step * dt;
        PetscReal time_np1 = (step + 1) * dt;

        if (run_type == RUN_MMS) {
            PetscScalar *f_local;
            DMDALocalInfo info;
            ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
            ierr = DMDAVecGetArray(da, f_source, &f_local); CHKERRQ(ierr);
            PetscReal f_time = (ts_type == TS_IMPLICIT) ? time_np1 : time_n;
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                PetscReal x = i * dx;
                f_local[i] = (diffusivity*PETSC_PI*PETSC_PI - 1.0) * sin(PETSC_PI*x) * exp(-f_time);
            }
            ierr = DMDAVecRestoreArray(da, f_source, &f_local); CHKERRQ(ierr);
        }

        if (ts_type == TS_IMPLICIT) {
            ierr = VecCopy(u, u_old); CHKERRQ(ierr);
            if (run_type == RUN_MMS) { ierr = VecAXPY(u_old, dt, f_source); CHKERRQ(ierr); }
            ierr = KSPSolve(ksp, u_old, u); CHKERRQ(ierr);
        } else { // Explicit
            ierr = MatMult(L, u, laplacian_u); CHKERRQ(ierr);
            ierr = VecAXPY(u, dt * diffusivity, laplacian_u); CHKERRQ(ierr);
            if (run_type == RUN_MMS) { ierr = VecAXPY(u, dt, f_source); CHKERRQ(ierr); }
            ierr = EnforceBoundaryConditions(da, u);CHKERRQ(ierr);
        }

        /* --- Full-featured Output Logic (Inside the loop) --- */
        if (checkpoint_interval > 0 && (step + 1) % checkpoint_interval == 0 && (step + 1) < max_steps) {
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
        PetscScalar *uexact_local;
        ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, u_exact, &uexact_local); CHKERRQ(ierr);
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            PetscReal x = i * dx;
            uexact_local[i] = sin(PETSC_PI * x) * exp(-final_time);
        }
        ierr = DMDAVecRestoreArray(da, u_exact, &uexact_local); CHKERRQ(ierr);
        
        PetscReal error_norm;
        ierr = VecAXPY(u, -1.0, u_exact); // u = u - u_exact
        ierr = VecNorm(u, NORM_INFINITY, &error_norm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------\n"); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MMS Verification Result (1D):\n"); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "L-infinity Error at T=%.2f with nx=%d, dt=%.6f is: %g\n", (double)final_time, nx, (double)dt, (double)error_norm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------\n"); CHKERRQ(ierr);
    }
    
    /* --- Final Output and Cleanup --- */
    PetscViewer vtk_viewer; char filename[PETSC_MAX_PATH_LEN];
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
