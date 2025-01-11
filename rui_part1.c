#include <stdio.h>        // needed for printing
#include <math.h>         // needed for tanh, used in init function
#include "params.h"       // model & simulation parameters
#include <omp.h>          // OpenMP header
#include <time.h>         // needed for measuring execution time

void init(double u[N][N], double v[N][N]) {
    #pragma omp parallel for collapse(2) schedule(dynamic) // Parallelize nested loops with dynamic scheduling
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((i - N / 2) / 16.0));
            v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
        }
    }
}

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    double lapu, lapv;
    int up, down, left, right;
    #pragma omp parallel for private(lapu, lapv, up, down, left, right) collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0) down = i; else down = i - 1;
            if (i == N - 1) up = i; else up = i + 1;
            if (j == 0) left = j; else left = j - 1;
            if (j == N - 1) right = j; else right = j + 1;

            lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] - 4.0 * u[i][j];
            lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] - 4.0 * v[i][j];
            du[i][j] = DD * lapu + f(u[i][j], v[i][j]) + R * stim(i, j);
            dv[i][j] = d * DD * lapv + g(u[i][j], v[i][j]);
        }
    }
}

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] += dt * du[i][j];
            v[i][j] += dt * dv[i][j];
        }
    }
}

double norm(double x[N][N]) {
    double nrmx = 0.0;

    #pragma omp parallel for reduction(+:nrmx) collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            nrmx += x[i][j] * x[i][j];
        }
    }
    return sqrt(nrmx);
}
int main(int argc, char** argv) {
    double t = 0.0, nrmu, nrmv;
    double u[N][N], v[N][N], du[N][N], dv[N][N];

    // Measure execution time
    clock_t start, end;
    double cpu_time_used;

    start = clock(); // Start timing

    FILE *fptr = fopen("nrms.txt", "w");
    if (fptr == NULL) {
        perror("Error opening file");
        return 1;
    }
    fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");

    // Initialize the state
    init(u, v);

    // Time-loop
    for (int k = 0; k < M; k++) {
        // Track the time
        t = dt * k;

        // Evaluate the PDE
        dxdt(du, dv, u, v);

        // Update the state variables u, v
        step(du, dv, u, v);

        if (k % m == 0) {
            // Calculate the norms
            nrmu = norm(u);
            nrmv = norm(v);
            printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
            fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
        }
    }

    fclose(fptr);

    end = clock(); // End timing

    // Calculate and print execution time
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel Execution Time: %f seconds\n", cpu_time_used);

    return 0;
}

