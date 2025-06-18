static char help[] = "Parallel 2D Heat Solver with switchable Explicit/Implicit methods and separate VTK Output files.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

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
    // PETSc and problem variables
    KSP                 ksp;
    Vec                 u, u_old;
    DM                  da;
    PetscErrorCode      ierr;
    TimeSteppingType    ts_type = TS_IMPLICIT;

    // Parameters
    PetscInt       nx = 50, ny = 50;
    PetscInt       max_steps = 200;
    PetscReal      final_time = 0.2;
    PetscReal      diffusivity = 0.01;
    PetscReal      dt = 0.0;

    ierr = PetscInitialize(&argc, &args, (char*)0, help); CHKERRQ(ierr);
    
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "Heat Equation Solver Options", "");
    ierr = PetscOptionsEnum("-ts_type", "Time stepping method", "main", TimeSteppingTypes, (PetscEnum)ts_type, (PetscEnum*)&ts_type, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nx", "Grid points in x direction", "main", nx, &nx, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ny", "Grid points in y direction", "main", ny, &ny, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-max_steps", "Number of time steps (used if -dt is not set)", "main", max_steps, &max_steps, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt", "Time step size (overrides max_steps)", "main", dt, &dt, NULL);CHKERRQ(ierr);
    PetscOptionsEnd();

    // Setup DMDA and Vectors
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                        nx, ny, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &u_old); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u, "Temperature"); CHKERRQ(ierr);

    // Set Initial Conditions
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

    // Calculate dt and max_steps based on user input
    if (dt > 0.0) {
        max_steps = (PetscInt)(final_time / dt);
    } else {
        dt = final_time / max_steps;
    }
    
    if (ts_type == TS_IMPLICIT) {
        // --- IMPLICIT METHOD ---
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Running with IMPLICIT method, dt = %g, steps = %d\n", (double)dt, max_steps);CHKERRQ(ierr);
        Mat A;
        ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
        MatStencil row, col[5]; PetscScalar v[5];
        PetscReal dx = 1.0 / (nx - 1); PetscReal dy = 1.0 / (ny - 1);
        PetscReal alpha_x = diffusivity * dt / (dx * dx);
        PetscReal alpha_y = diffusivity * dt / (dy * dy);
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                row.j = j; row.i = i;
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                    v[0] = 1.0; ierr = MatSetValuesStencil(A, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
                } else {
                    v[0] = 1.0 + 2.0 * alpha_x + 2.0 * alpha_y; col[0] = row;
                    v[1] = -alpha_x; col[1].j = j; col[1].i = i-1;
                    v[2] = -alpha_x; col[2].j = j; col[2].i = i+1;
                    v[3] = -alpha_y; col[3].j = j-1; col[3].i = i;
                    v[4] = -alpha_y; col[4].j = j+1; col[4].i = i;
                    ierr = MatSetValuesStencil(A, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

        for (PetscInt step = 0; step < max_steps; ++step) {
            ierr = VecCopy(u, u_old); CHKERRQ(ierr);
            ierr = KSPSolve(ksp, u_old, u); CHKERRQ(ierr);
        }
        ierr = MatDestroy(&A); CHKERRQ(ierr);
        ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

    } else {
        // --- EXPLICIT METHOD ---
        PetscReal dx = 1.0 / (nx - 1); PetscReal dy = 1.0 / (ny - 1);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Running with EXPLICIT method, dt = %g, steps = %d\n", (double)dt, max_steps);CHKERRQ(ierr);
        
        PetscReal stability_factor = diffusivity * dt * (1.0/(dx*dx) + 1.0/(dy*dy));
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Stability Factor: %g (must be <= 0.5)\n", (double)stability_factor); CHKERRQ(ierr);
        if (stability_factor > 0.5) {
            ierr = PetscPrintf(PETSC_COMM_WORLD, "WARNING: Stability condition not met! Solution may diverge.\n");CHKERRQ(ierr);
        }

        Mat L; Vec laplacian_u;
        ierr = DMCreateMatrix(da, &L); CHKERRQ(ierr);
        ierr = VecDuplicate(u, &laplacian_u); CHKERRQ(ierr);

        MatStencil row, col[5]; PetscScalar v[5];
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                row.j = j; row.i = i;
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                    v[0] = 0.0; ierr = MatSetValuesStencil(L, 1, &row, 1, &row, v, INSERT_VALUES); CHKERRQ(ierr);
                } else {
                    v[0] = -2.0/(dx*dx) - 2.0/(dy*dy); col[0] = row;
                    v[1] = 1.0/(dx*dx); col[1].j = j; col[1].i = i-1;
                    v[2] = 1.0/(dx*dx); col[2].j = j; col[2].i = i+1;
                    v[3] = 1.0/(dy*dy); col[3].j = j-1; col[3].i = i;
                    v[4] = 1.0/(dy*dy); col[4].j = j+1; col[4].i = i;
                    ierr = MatSetValuesStencil(L, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); ierr = MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        
        for (PetscInt step = 0; step < max_steps; ++step) {
            ierr = MatMult(L, u, laplacian_u); CHKERRQ(ierr);
            ierr = VecAXPY(u, dt * diffusivity, laplacian_u); CHKERRQ(ierr);
            ierr = EnforceBoundaryConditions(da, u);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&L); CHKERRQ(ierr);
        ierr = VecDestroy(&laplacian_u); CHKERRQ(ierr);
    }
    
    // MODIFIED: Save final solution with a different name based on the method
    char        filename[PETSC_MAX_PATH_LEN];
    PetscViewer vtk_viewer;

    if (ts_type == TS_IMPLICIT) {
        ierr = PetscSNPrintf(filename, sizeof(filename), "implicit_solution.vts"); CHKERRQ(ierr);
    } else {
        ierr = PetscSNPrintf(filename, sizeof(filename), "explicit_solution.vts"); CHKERRQ(ierr);
    }
    
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Writing final solution to %s\n", filename); CHKERRQ(ierr);

    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &vtk_viewer); CHKERRQ(ierr);
    ierr = VecView(u, vtk_viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtk_viewer); CHKERRQ(ierr);

    // Clean up
    ierr = VecDestroy(&u); CHKERRQ(ierr); ierr = VecDestroy(&u_old); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}