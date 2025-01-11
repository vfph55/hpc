#include <stdio.h>               // For printing
#include <math.h>                // For tanh function
#include <mpi.h>                 // MPI header
#include "params.h"              // Model & simulation parameters

// Initialization function
void init(double u[N][N], double v[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((i - N / 2) / 16.0));
            v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
        }
    }
}

// Compute derivatives
void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_process;

    // Exchange ghost rows
    if (rank > 0)
        MPI_Send(u[start_row], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Send(u[end_row - 1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Recv(u[end_row], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank > 0)
        MPI_Recv(u[start_row - 1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank > 0)
        MPI_Send(v[start_row], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Send(v[end_row - 1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Recv(v[end_row], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank > 0)
        MPI_Recv(v[start_row - 1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Compute derivatives
    for (int i = start_row; i < end_row; i++) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_process;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] += dt * du[i][j];
            v[i][j] += dt * dv[i][j];
        }
    }
}

// Calculate norm
double norm(double x[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_process;

    double local_sum = 0.0;
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            local_sum += x[i][j] * x[i][j];
        }
    }

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

// Main function
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double u[N][N], v[N][N], du[N][N], dv[N][N];
    double start_time, end_time, t = 0.0;

    FILE *fptr = NULL;
    if (rank == 0) {
        printf("Starting simulation with %d processes...\n", size);
        fptr = fopen("part2.dat", "w");
        fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");
    }

    start_time = MPI_Wtime();
    init(u, v);

    for (int k = 0; k < M; k++) {
        t = dt * k;
        dxdt(du, dv, u, v);
        step(du, dv, u, v);

        if (k % m == 0) {
            double nrmu = norm(u);
            double nrmv = norm(v);
            if (rank == 0) {
                printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
                fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
            }
        }
    }

    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Total simulation time: %f seconds\n", end_time - start_time);
        fclose(fptr);
    }

    MPI_Finalize();
    return 0;
}
