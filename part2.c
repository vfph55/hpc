#include <stdio.h>               // needed for printing
#include <math.h>               // needed for tanh, used in init function
#include "params.h"             // model & simulation parameters
#include <mpi.h>                // MPI header

void init(double u[N][N], double v[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size; // Determine local rows per process
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            int global_i = rank * rows_per_process + i; // Global index for rows
            u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((global_i - N / 2) / 16.0));
            v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
        }
    }
}

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    double lapu, lapv;

    // Boundary exchange to handle array boundaries
    if (rank != 0) {
        MPI_Send(u[0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    }
    if (rank != size - 1) {
        MPI_Send(u[rows_per_process - 1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }
    
    if (rank != 0) {
        MPI_Recv(u[-1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank != size - 1) {
        MPI_Recv(u[rows_per_process], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            int global_i = rank * rows_per_process + i;

            // Determine neighboring indices, handling boundaries accordingly
            int up = (i == rows_per_process - 1) ? i : i + 1;      // Local up
            int down = (i == 0) ? 0 : i - 1;                       // Local down
            int left = (j == 0) ? 0 : j - 1;                       // Local left
            int right = (j == N - 1) ? N - 1 : j + 1;             // Local right

            lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] - 4.0 * u[i][j];
            lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] - 4.0 * v[i][j];
            du[i][j] = DD * lapu + f(u[i][j], v[i][j]) + R * stim(global_i, j);
            dv[i][j] = d * DD * lapv + g(u[i][j], v[i][j]);
        }
    }
}

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    int rows_per_process, i, j;
    MPI_Comm_size(MPI_COMM_WORLD, &rows_per_process);
    
    for (i = 0; i < N / rows_per_process; i++) {
        for (j = 0; j < N; j++) {
            u[i][j] += dt * du[i][j];
            v[i][j] += dt * dv[i][j];
        }
    }
}

double norm(double x[N][N]) {
    double local_nrm = 0.0;
    int rows_per_process;
    MPI_Comm_size(MPI_COMM_WORLD, &rows_per_process);
    for (int i = 0; i < N / rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_nrm += x[i][j] * x[i][j];
        }
    }

    // Use MPI_Reduce to sum up the norms from all processes
    double global_nrm = 0.0;
    MPI_Reduce(&local_nrm, &global_nrm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_nrm;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); // Initialize MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process ID

    double t = 0.0;
    double u[N][N], v[N][N], du[N][N], dv[N][N];

    // Initialize state
    init(u, v);

    // Time-loop
    for (int k = 0; k < M; k++) {
        // Track the time
        t = dt * k;

        // Evaluate the PDE
        dxdt(du, dv, u, v);

        // Update the state variables u and v
        step(du, dv, u, v);

        // Calculate the norms at a specific interval (if rank 0)
        if (k % m == 0 && rank == 0) {
            double nrmu = norm(u);
            double nrmv = norm(v);
            printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}