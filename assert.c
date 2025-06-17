#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h> // 引入断言库

/**
 * @brief  为一个二维网格分配连续的内存空间
 * @param  nx   x方向的网格点数
 * @param  ny   y方向的网格点数
 * @return      成功则返回指向二维数组的指针，失败则返回NULL
 */
double** allocate_grid(int nx, int ny) {

    if (nx <= 0 || ny <= 0) {
        fprintf(stderr, "Error in allocate_grid: Grid dimensions must be positive.\n");
        return NULL;
    }

    // `sizeof(double*)` 保证了指针数组的正确大小
    double** grid = (double**)malloc(nx * sizeof(double*));
    

    if (grid == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for grid pointers.\n");
        return NULL;
    }
    

    double* storage = (double*)calloc((size_t)nx * ny, sizeof(double));
    

    if (storage == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for grid data storage.\n");
        free(grid); // 清理之前分配的内存
        return NULL;
    }


    for (int i = 0; i < nx; i++) {
        grid[i] = &storage[(size_t)i * ny];
    }
    return grid;
}

/**
 * @brief  释放二维网格的内存
 * @param  grid 指向二维网格的指针
 */
void free_grid(double** grid) {

    if (grid != NULL) {
        // 首先释放连续的数据块
        free(grid[0]);
        // 然后释放指针数组
        free(grid);
    }
}

int main(int argc, char *argv[]) {
    // ====== 1. Problem Configuration ======
    const double L = 1.0;
    const double final_time = 0.1;
    const double diffusivity = 0.01;

    const int NX = 51;
    const int NY = 51;


    assert(NX > 2 && NY > 2 && "Grid dimensions must be large enough for boundaries.");

    const double dx = L / (NX - 1);
    const double dy = L / (NY - 1);
    const double dt = 0.0001;
    
    // ====== 2. Stability and Parameter Check ======
    double stability_factor = diffusivity * dt * (1.0/(dx*dx) + 1.0/(dy*dy));
    printf("Stability factor: %f\n", stability_factor);
    if (stability_factor > 0.5) {
        fprintf(stderr, "WARNING: Stability condition not met! Result may be unstable.\n");
    }

    // ====== 3. Memory Allocation ======
    double** u = allocate_grid(NX, NY);
    double** u_new = allocate_grid(NX, NY);


    if (u == NULL || u_new == NULL) {
        fprintf(stderr, "Critical Error in main: Failed to allocate memory for grids. Exiting.\n");
        // 确保在退出前释放已成功分配的内存
        free_grid(u);
        free_grid(u_new);
        return 1;
    }

    // ====== 4. Initialization (Initial and Boundary Conditions) ======
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
    
    // ====== 5. Time Marching Loop ======
    double time = 0.0;
    while (time < final_time) {

        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {

                assert(i > 0 && i < NX-1 && j > 0 && j < NY-1);

                double u_xx = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / (dx * dx);
                double u_yy = (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / (dy * dy);
                u_new[i][j] = u[i][j] + diffusivity * dt * (u_xx + u_yy);
            }
        }
        

        double** temp = u;
        u = u_new;
        u_new = temp;

        time += dt;
    }

    // ====== 6. Output and Cleanup ======
    FILE *fp = fopen("output.dat", "w");

    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open output file 'output.dat'.\n");
        free_grid(u);
        free_grid(u_new);
        return 1;
    }

    fprintf(fp, "# x y u\n");
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(fp, "%f %f %f\n", i * dx, j * dy, u[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Simulation finished. Final temperature distribution saved to output.dat\n");

    // 释放内存
    free_grid(u);
    free_grid(u_new);

    return 0;
}