static char help[] = "Parallel 2D Transient Heat Equation solver using PETSc.\n\n";

#include <petscksp.h>

// Function to save the solution vector to a file for visualization
PetscErrorCode SaveSolution(Vec u, PetscInt nx, PetscInt ny, const char* filename);

int main(int argc, char **args) {
    Vec            u, u_old;      // u: current solution, u_old: solution from previous step (RHS)
    Mat            A;             // System matrix for the implicit method
    KSP            ksp;           // Linear solver context
    PetscInt       nx = 50, ny = 50; // Grid dimensions
    PetscInt       N;             // Total number of grid points
    PetscInt       max_steps = 100; // Maximum time steps
    PetscReal      final_time = 0.1;
    PetscReal      diffusivity = 0.01;
    PetscReal      dx, dy, dt;
    PetscInt       i, j, row, col[5];
    PetscScalar    value[5];
    PetscInt       rstart, rend, its;

    PetscInitialize(&argc, &args, (char*)0, help);
    PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL);
    PetscOptionsGetInt(NULL, NULL, "-max_steps", &max_steps, NULL);

    N = nx * ny; // Total degrees of freedom
    dx = 1.0 / (nx - 1);
    dy = 1.0 / (ny - 1);
    dt = final_time / max_steps;

    // --- 1. Create and configure vectors ---
    // u will hold the solution at the current time step
    // u_old will hold the solution from the previous time step, serving as the RHS
    VecCreate(PETSC_COMM_WORLD, &u);
    VecSetSizes(u, PETSC_DECIDE, N);
    VecSetFromOptions(u);
    VecDuplicate(u, &u_old);

    // --- 2. Create and assemble the matrix A for the implicit system ---
    // The equation is u_new - D*dt*Laplacian(u_new) = u_old
    // So the matrix A is (I - D*dt*Laplacian)
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);
    MatSetUp(A);

    // Get the range of rows that this process owns
    MatGetOwnershipRange(A, &rstart, &rend);

    PetscReal alpha_x = diffusivity * dt / (dx * dx);
    PetscReal alpha_y = diffusivity * dt / (dy * dy);

    // Loop over the rows owned by this process to assemble the matrix
    for (row = rstart; row < rend; ++row) {
        // Map the 1D row index back to 2D grid coordinates (i, j)
        i = row % nx; // x-index
        j = row / nx; // y-index

        // Check for boundary nodes. For boundary nodes, we enforce u=0.
        // We do this by setting the diagonal element to 1 and all other elements in that row to 0.
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
            value[0] = 1.0;
            MatSetValues(A, 1, &row, 1, &row, value, INSERT_VALUES);
        } else {
            // Interior node: assemble the 5-point stencil for (I - D*dt*Laplacian)
            // Diagonal element
            col[0] = row;
            value[0] = 1.0 + 2.0 * alpha_x + 2.0 * alpha_y;
            // West neighbor
            col[1] = row - 1;
            value[1] = -alpha_x;
            // East neighbor
            col[2] = row + 1;
            value[2] = -alpha_x;
            // South neighbor
            col[3] = row - nx;
            value[3] = -alpha_y;
            // North neighbor
            col[4] = row + nx;
            value[4] = -alpha_y;
            MatSetValues(A, 1, &row, 5, col, value, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    // --- 3. Set Initial Conditions ---
    // A "hot spot" in the center of the domain
    PetscScalar *u_array;
    VecGetArray(u, &u_array);
    for (row = rstart; row < rend; ++row) {
        i = row % nx;
        j = row / nx;
        PetscReal x_coord = i * dx;
        PetscReal y_coord = j * dy;
        if (x_coord >= 0.4 && x_coord <= 0.6 && y_coord >= 0.4 && y_coord <= 0.6) {
            u_array[row - rstart] = 100.0;
        } else {
            u_array[row - rstart] = 0.0;
        }
        // Enforce boundary condition u=0
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
             u_array[row - rstart] = 0.0;
        }
    }
    VecRestoreArray(u, &u_array);
    
    // --- 4. Setup the KSP Solver ---
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp); // Allow runtime customization

    // --- 5. Time Marching Loop ---
    PetscPrintf(PETSC_COMM_WORLD, "Starting time marching loop...\n");
    for (PetscInt step = 0; step < max_steps; ++step) {
        // The right-hand side is the solution from the previous step
        VecCopy(u, u_old);

        // Solve the linear system A * u = u_old for the current step's solution u
        KSPSolve(ksp, u_old, u);
        
        KSPGetIterationNumber(ksp, &its);
        PetscPrintf(PETSC_COMM_WORLD, "Time step %d finished after %d KSP iterations.\n", step + 1, its);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Simulation finished.\n");

    // --- 6. Save the final solution and Clean up ---
    SaveSolution(u, nx, ny, "final_solution.dat");

    VecDestroy(&u);
    VecDestroy(&u_old);
    MatDestroy(&A);
    KSPDestroy(&ksp);
    PetscFinalize();
    return 0;
}

// This helper function gathers the distributed vector u onto process 0
// and then writes it to a file in a format gnuplot can read.
PetscErrorCode SaveSolution(Vec u, PetscInt nx, PetscInt ny, const char* filename) {
    PetscErrorCode ierr;
    PetscViewer    viewer;
    Vec            u_global = NULL;
    PetscMPIInt    rank;
    VecScatter     ctx;

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    // 创建一个 VecScatter 上下文，用于将数据收集到0号进程
    ierr = VecScatterCreateToZero(u, &ctx, &u_global); CHKERRQ(ierr);

    // 执行数据收集操作
    ierr = VecScatterBegin(ctx, u, u_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx, u, u_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    
    // 销毁上下文
    ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);

    // 只有0号进程负责将收集到的数据写入文件
    if (rank == 0) {
        ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        ierr = VecView(u_global, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    }
    
    ierr = VecDestroy(&u_global); CHKERRQ(ierr);
    return 0;
}
