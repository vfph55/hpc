#include <stdio.h>               // needed for printing
#include <math.h>               // needed for tanh, used in init function
#include "params.h"             // model & simulation parameters
#include <mpi.h>                // MPI header

void init(double u[N][N], double v[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] = ulo + (uhi-ulo)*0.5*(1.0 + tanh((i-N/2)/16.0));
            v[i][j] = vlo + (vhi-vlo)*0.5*(1.0 + tanh((j-N/2)/16.0));
        }
    }
}

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_process;

    double lapu, lapv;
    int up, down, left, right;

    // Exchange ghost rows
    if (rank > 0)
        MPI_Send(u[start_row], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Send(u[end_row-1], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Recv(u[end_row], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank > 0)
        MPI_Recv(u[start_row-1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Same for v
    if (rank > 0)
        MPI_Send(v[start_row], N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Send(v[end_row-1], N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);
    if (rank < size - 1)
        MPI_Recv(v[end_row], N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank > 0)
        MPI_Recv(v[start_row-1], N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            down = (i == 0) ? i : i-1;
            up = (i == N-1) ? i : i+1;
            left = (j == 0) ? j : j-1;
            right = (j == N-1) ? j : j+1;

            lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] - 4.0*u[i][j];
            lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] - 4.0*v[i][j];
            du[i][j] = DD*lapu + f(u[i][j], v[i][j]) + R*stim(i,j);
            dv[i][j] = d*DD*lapv + g(u[i][j], v[i][j]);
        }
    }
}

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_process;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] += dt*du[i][j];
            v[i][j] += dt*dv[i][j];
        }
    }
}

double norm(double x[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_process;

    double local_nrmx = 0.0;
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            local_nrmx += x[i][j]*x[i][j];
        }
    }

    double global_nrmx;
    MPI_Allreduce(&local_nrmx, &global_nrmx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_nrmx;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    start_time = MPI_Wtime();

    double t = 0.0, nrmu, nrmv;
    double u[N][N], v[N][N], du[N][N], dv[N][N];

    FILE *fptr = NULL;
    if (rank == 0) {
        fptr = fopen("part2.dat", "w");
        fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");
    }

    // initialize the state
    init(u, v);

    // time-loop
    for (int k = 0; k < M; k++) {
        // track the time
        t = dt*k;
        // evaluate the PDE
        dxdt(du, dv, u, v);
        // update the state variables u,v
        step(du, dv, u, v);
        if (k%m == 0) {
            // calculate the norms
            nrmu = norm(u);
            nrmv = norm(v);
            if (rank == 0) {
                printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
                fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
            }
        }
    }

    if (rank == 0) {
        fclose(fptr);
    }

    end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    double max_total_time;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total execution time: %f seconds\n", max_total_time);
    }

    MPI_Finalize();
    return 0;
}