#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function to allocate a 2D grid
double** allocate_grid(int nx, int ny) {
    double** grid = (double**)malloc(nx * sizeof(double*));
    if (grid == NULL) return NULL;
    
    double* storage = (double*)calloc(nx * ny, sizeof(double));
    if (storage == NULL) {
        free(grid);
        return NULL;
    }

    for (int i = 0; i < nx; i++) {
        grid[i] = &storage[i * ny];
    }
    return grid;
}

// Helper function to free the 2D grid
void free_grid(double** grid) {
    if (grid != NULL) {
        free(grid[0]); // Free the contiguous storage block
        free(grid);    // Free the array of pointers
    }
}

int main() {
    // 1. Problem Configuration
    // ----------------------------------------------------------------
    // Physical and domain parameters
    const double L = 1.0;            // Domain size (unit square) 
    const double final_time = 0.1;   // Final simulation time
    const double diffusivity = 0.01; // Thermal diffusivity (kappa / (rho * c))

    // Discretization parameters
    const int NX = 51;               // Number of points in x-direction
    const int NY = 51;               // Number of points in y-direction
    const double dx = L / (NX - 1);  // Spatial step size 
    const double dy = L / (NY - 1);  // Spatial step size 
    const double dt = 0.0001;        // Time step size 

    // Stability check for Explicit Euler
    // For 2D, we need D * dt * (1/dx^2 + 1/dy^2) <= 0.5
    // This is a key part of the numerical analysis required.
    double stability_factor = diffusivity * dt * (1.0/(dx*dx) + 1.0/(dy*dy));
    printf("========================================\n");
    printf("2D Heat Equation Solver: Explicit Euler\n");
    printf("Domain size: %.2f x %.2f\n", L, L);
    printf("Grid size: %d x %d\n", NX, NY);
    printf("Spatial step (dx, dy): %f\n", dx);
    printf("Time step (dt): %f\n", dt);
    printf("Stability factor: %f\n", stability_factor);
    if (stability_factor > 0.5) {
        printf("WARNING: Stability condition not met! Result may be unstable.\n");
    }
    printf("========================================\n\n");

    // 2. Memory Allocation
    // ----------------------------------------------------------------
    // We need two grids: one for the current time step (u) and one for the next (u_new)
    double** u = allocate_grid(NX, NY);
    double** u_new = allocate_grid(NX, NY);

    if (u == NULL || u_new == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for grids.\n");
        return 1;
    }

    // 3. Initialization (Initial and Boundary Conditions)
    // ----------------------------------------------------------------
    // Initial condition: A 'hot' square in the center, zero elsewhere.
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x = i * dx;
            double y = j * dy;
            if (x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6) {
                u[i][j] = 100.0;
            } else {
                u[i][j] = 0.0;
            }
        }
    }
    
    // Boundary conditions: Hold boundaries at a constant temperature (e.g., 0).
    // With calloc, they are already 0, but we enforce it for clarity.
    // u_new will also have its boundaries set to 0.
    for (int i = 0; i < NX; i++) {
        u_new[i][0] = 0.0;
        u_new[i][NY-1] = 0.0;
    }
    for (int j = 0; j < NY; j++) {
        u_new[0][j] = 0.0;
        u_new[NX-1][j] = 0.0;
    }
    
    // 4. Time Marching Loop
    // ----------------------------------------------------------------
    double time = 0.0;
    int iter = 0;
    while (time < final_time) {
        // Copy boundary conditions from u to u_new
        for (int i = 0; i < NX; i++) {
             u_new[i][0] = u[i][0];
             u_new[i][NY-1] = u[i][NY-1];
        }
        for (int j = 0; j < NY; j++) {
            u_new[0][j] = u[0][j];
            u_new[NX-1][j] = u[NX-1][j];
        }

        // Apply the Finite Difference stencil (Explicit Euler)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                double u_xx = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / (dx * dx);
                double u_yy = (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / (dy * dy);
                u_new[i][j] = u[i][j] + diffusivity * dt * (u_xx + u_yy);
            }
        }
        
        // Swap grids for the next iteration by swapping pointers
        double** temp = u;
        u = u_new;
        u_new = temp;

        time += dt;
        iter++;
        if (iter % 100 == 0) {
            printf("Iteration %d, Time = %f\n", iter, time);
        }
    }

    // 5. Output and Cleanup
    // ----------------------------------------------------------------
    printf("\nSimulation finished at time = %f\n", time);
    FILE *fp = fopen("output.dat", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file.\n");
        free_grid(u);
        free_grid(u_new);
        return 1;
    }

    fprintf(fp, "# x y u\n"); // Header for gnuplot
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fp, "%f %f %f\n", i * dx, j * dy, u[i][j]);
        }
        fprintf(fp, "\n"); // Newline for gnuplot pm3d map
    }
    fclose(fp);
    printf("Final temperature distribution saved to output.dat\n");

    // Free allocated memory
    free_grid(u);
    free_grid(u_new);

    return 0;
}