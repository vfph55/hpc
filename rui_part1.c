#include <stdio.h>              // For printing
#include <math.h>               // For tanh function
#include <omp.h>                // For OpenMP
#include "params.h"             // Model & simulation parameters

// Initialization function
void init(double u[N][N], double v[N][N]) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((i - N / 2.0) / 16.0));
            v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2.0) / 16.0));
        }
    }
}

// Compute derivatives
void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int up = (i == N - 1) ? i : i + 1;
            int down = (i == 0) ? i : i - 1;
            int left = (j == 0) ? j : j - 1;
            int right = (j == N - 1) ? j : j + 1;

            double lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] - 4.0 * u[i][j];
            double lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] - 4.0 * v[i][j];

            du[i][j] = DD * lapu + f(u[i][j], v[i][j]) + R * stim(i, j);
            dv[i][j] = d * DD * lapv + g(u[i][j], v[i][j]);
        }
    }
}

// Update state variables
void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] += dt * du[i][j];
            v[i][j] += dt * dv[i][j];
        }
    }
}

// Calculate norm
double norm(double x[N][N]) {
    double nrmx = 0.0;
    #pragma omp parallel for reduction(+:nrmx)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            nrmx += x[i][j] * x[i][j];
        }
    }
    return nrmx;
}
// Main function
int main(int argc, char** argv) {
    double total_times[4];

   

        double start_time = omp_get_wtime();
        double u[N][N], v[N][N], du[N][N], dv[N][N];
        double t = 0.0, nrmu, nrmv;

        FILE* fptr = fopen("part1.dat", "w");
        fprintf(fptr, "#t\tnrmu\tnrmv\n");

        init(u, v);

        for (int k = 0; k < M; k++) {
            t = dt * k;
            dxdt(du, dv, u, v);
            step(du, dv, u, v);

            if (k % m == 0) {
                nrmu = norm(u);
                nrmv = norm(v);
                printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
                fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
            }
        }

        fclose(fptr);

        double end_time = omp_get_wtime();
        double total_time = end_time - start_time;
        printf("Time: %f\n", total_time);
    

    return 0;
}