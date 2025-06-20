static char help[] = "Final Corrected Parallel 2D Heat Solver with HDF5 Restart and switchable solvers.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscviewerhdf5.h>

// Define an enum for our time-stepping methods
typedef enum {TS_IMPLICIT, TS_EXPLICIT} TimeSteppingType;
const char *const TimeSteppingTypes[] = {"implicit", "explicit", "TimeSteppingType", "TS_", 0};

// Helper function to enforce boundary conditions on a vector
PetscErrorCode EnforceBoundaryConditions(DM da, Vec u)
{
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscScalar    **u_local;

    ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, u, &u_local);CHKERRQ(ierr);
    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
        for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
                u_local[j][i] = 0.0;
            }
        }
    }
    ierr = DMDAVecRestoreArray(da, u, &u_local);CHKERRQ(ierr);
    return 0;
}

int main(int argc, char **args) {
    /* --- Variable Declarations --- */
    DM                  da;
    Vec                 u, u_old;
    Mat                 A = NULL, L = NULL;
    KSP                 ksp;
    PetscErrorCode      ierr;
    TimeSteppingType    ts_type = TS_IMPLICIT;
    PetscInt            nx = 50, ny = 50, max_steps = 200, start_step = 0;
    PetscReal           final_time = 0.4, diffusivity = 0.01, dt = 0.0;
    PetscBool           flg_restart = PETSC_FALSE;
    char                checkpoint_file[] = "checkpoint.h5";
    PetscInt            checkpoint_interval = 10, vtk_interval = 20;
    Vec                 laplacian_u = NULL;

    /* --- Initialization and Option Parsing --- */
    ierr = PetscInitialize(&argc, &args, (char*)0, help); CHKERRQ(ierr);
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "Heat Equation Solver Options", "");
    ierr = PetscOptionsEnum("-ts_type", "Time stepping method", "main", TimeSteppingTypes, (PetscEnum)ts_type, (PetscEnum*)&ts_type, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nx", "Grid points in x direction", "main", nx, &nx, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ny", "Grid points in y direction", "main", ny, &ny, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-max_steps", "Number of time steps (used if -dt is not set)", "main", max_steps, &max_steps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt", "Time step size (overrides max_steps)", "main", dt, &dt, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-restart", "Enable restarting from checkpoint file", "main", flg_restart, &flg_restart, NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    /* --- DMDA and Vector Setup --- */
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u, "Temperature"); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &u_old); CHKERRQ(ierr);

    /* --- Initial State: Load from checkpoint or set Initial Conditions --- */
    if (flg_restart) {
        PetscViewer hdf5_viewer;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Attempting to restart from %s\n", checkpoint_file); CHKERRQ(ierr);
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, checkpoint_file, FILE_MODE_READ, &hdf5_viewer); CHKERRQ(ierr);
        ierr = VecLoad(u, hdf5_viewer); CHKERRQ(ierr);
        Vec v_step;
        ierr = VecCreate(PETSC_COMM_WORLD, &v_step); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)v_step, "step"); CHKERRQ(ierr);
        ierr = VecLoad(v_step, hdf5_viewer); CHKERRQ(ierr);

        // --- FINAL FIX: Only process 0 reads the value, then broadcasts it to all others ---
        PetscMPIInt rank;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
        if (rank == 0) {
            const PetscScalar *step_array;
            ierr = VecGetArrayRead(v_step, &step_array); CHKERRQ(ierr);
            start_step = (PetscInt)PetscRealPart(step_array[0]);
            ierr = VecRestoreArrayRead(v_step, &step_array); CHKERRQ(ierr);
        }
        // Broadcast the start_step from process 0 to all other processes
        ierr = MPI_Bcast(&start_step, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        
        ierr = VecDestroy(&v_step); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&hdf5_viewer); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Successfully restarted from step %d\n", start_step); CHKERRQ(ierr);
    } else {
        DMDALocalInfo info;
        PetscScalar   **u_local;
        ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da, u, &u_local); CHKERRQ(ierr);
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                if (i >= nx * 0.4 && i <= nx * 0.6 && j >= ny * 0.4 && j <= ny * 0.6) u_local[j][i] = 100.0;
                else u_local[j][i] = 0.0;
            }
        }
        ierr = DMDAVecRestoreArray(da, u, &u_local); CHKERRQ(ierr);
        ierr = EnforceBoundaryConditions(da, u); CHKERRQ(ierr);
    }

    /* --- Pre-Loop Setup --- */
    if (dt > 0.0) { max_steps = (PetscInt)(final_time / dt); } else { dt = final_time / max_steps; }
    
    if (ts_type == TS_IMPLICIT) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting up for IMPLICIT method, dt = %g, steps = %d\n", (double)dt, max_steps);CHKERRQ(ierr);
        ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
        DMDALocalInfo info; ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        MatStencil row, col[5]; PetscScalar v[5];
        PetscReal dx = 1.0/(nx-1), dy = 1.0/(ny-1), alpha_x = diffusivity*dt/(dx*dx), alpha_y = diffusivity*dt/(dy*dy);
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                row.j = j; row.i = i;
                if (i==0 || i==nx-1 || j==0 || j==ny-1) { v[0] = 1.0; ierr = MatSetValuesStencil(A, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr); }
                else {
                    v[0]=1.0+2.0*alpha_x+2.0*alpha_y; col[0]=row; v[1]=-alpha_x; col[1].j=j; col[1].i=i-1; v[2]=-alpha_x; col[2].j=j; col[2].i=i+1;
                    v[3]=-alpha_y; col[3].j=j-1; col[3].i=i; v[4]=-alpha_y; col[4].j=j+1; col[4].i=i;
                    ierr = MatSetValuesStencil(A, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    } else { // Explicit
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Setting up for EXPLICIT method, dt = %g, steps = %d\n", (double)dt, max_steps);CHKERRQ(ierr);
        PetscReal dx = 1.0/(nx-1), dy = 1.0/(ny-1);
        PetscReal stability_factor = diffusivity*dt*(1.0/(dx*dx)+1.0/(dy*dy));
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Stability Factor: %g (must be <= 0.5)\n", (double)stability_factor); CHKERRQ(ierr);
        if (stability_factor > 0.5) { ierr = PetscPrintf(PETSC_COMM_WORLD, "WARNING: Stability condition not met!\n");CHKERRQ(ierr); }
        ierr = DMCreateMatrix(da, &L); CHKERRQ(ierr);
        ierr = VecDuplicate(u, &laplacian_u); CHKERRQ(ierr);
        DMDALocalInfo info; ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
        MatStencil row, col[5]; PetscScalar v[5];
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                row.j = j; row.i = i;
                if (i==0 || i==nx-1 || j==0 || j==ny-1) { v[0] = 0.0; ierr = MatSetValuesStencil(L, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr); }
                else {
                    v[0]=-2.0/(dx*dx)-2.0/(dy*dy); col[0]=row; v[1]=1.0/(dx*dx); col[1].j=j; col[1].i=i-1; v[2]=1.0/(dx*dx); col[2].j=j; col[2].i=i+1;
                    v[3]=1.0/(dy*dy); col[3].j=j-1; col[3].i=i; v[4]=1.0/(dy*dy); col[4].j=j+1; col[4].i=i;
                    ierr = MatSetValuesStencil(L, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    
    /* --- Main Time-Stepping Loop --- */
    for (PetscInt step = start_step; step < max_steps; ++step) {
        if (ts_type == TS_IMPLICIT) {
            ierr = VecCopy(u, u_old); CHKERRQ(ierr);
            ierr = KSPSolve(ksp, u_old, u); CHKERRQ(ierr);
        } else { // Explicit
            ierr = MatMult(L, u, laplacian_u); CHKERRQ(ierr);
            ierr = VecAXPY(u, dt * diffusivity, laplacian_u); CHKERRQ(ierr);
            ierr = EnforceBoundaryConditions(da, u);CHKERRQ(ierr);
        }

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
        if ((step + 1) % vtk_interval == 0 || (step + 1) == max_steps) {
             PetscViewer vtk_viewer; char filename[PETSC_MAX_PATH_LEN];
             ierr = PetscSNPrintf(filename, sizeof(filename), "solution-%04ld.vts", (long)(step + 1)); CHKERRQ(ierr);
             ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &vtk_viewer); CHKERRQ(ierr);
             ierr = VecView(u, vtk_viewer); CHKERRQ(ierr);
             ierr = PetscViewerDestroy(&vtk_viewer); CHKERRQ(ierr);
        }
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
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
