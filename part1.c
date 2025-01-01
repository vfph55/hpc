#include <stdio.h>              // needed for printing
#include <math.h>              // needed for tanh, used in init function
#include "params.h"            // model & simulation parameters
#include <omp.h>

void init(double u[N][N], double v[N][N]) {
    #pragma omp parallel for schedule(dynamic) // Static scheduling for predictable workload
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((i - N / 2) / 16.0));
            v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
        }
    }
}

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    double lapu, lapv; // Laplacian variables are defined outside the loop
    int up, down, left, right; // Neighbor indices
    
    // Parallelize the computation of dxdt using OpenMP
    #pragma omp parallel for private(lapu, lapv, up, down, left, right) // Use private to avoid race condition
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Determine neighboring indices
            down = (i == 0) ? i : i - 1;
            up = (i == N - 1) ? i : i + 1;
            left = (j == 0) ? j : j - 1;
            right = (j == N - 1) ? j : j + 1;

            up = (up<0) ? 0 : up;
            up = (up>N-1) ? N-1 :up;

            down = (down<0) ? 0 : down;
            down = (down>N-1) ? N-1 :down;

            left = (left<0) ? 0 : left;
            left = (left>N-1) ? N-1 :left;

            right = (right<0) ? 0 : right;
            right = (right>N-1) ? N-1 :right;


            // Calculate Laplacian values
            lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] - 4.0 * u[i][j];
            lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] - 4.0 * v[i][j];

            // Update derivatives
            du[i][j] = DD * lapu + f(u[i][j], v[i][j]) + R * stim(i, j);
            dv[i][j] = d * DD * lapv + g(u[i][j], v[i][j]);
        }
    }
}

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    // Parallelize the update step using OpenMP
    #pragma omp parallel for collapse(2) // Collapse to parallelize both loops
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i][j] += dt * du[i][j]; // Update u
            v[i][j] += dt * dv[i][j]; // Update v
        }
    }
}

double norm(double x[N][N]) {
    double nrmx = 0.0;
    // Using reduction to avoid atomic and improve performance
    #pragma omp parallel for reduction(+:nrmx) schedule(dynamic) // Use dynamic scheduling for better load balancing
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            nrmx += x[i][j] * x[i][j];
        }
    }
    return nrmx;
}

int main(int argc, char** argv){
    int n[] = {1,2,4,8};
    double total_times[4];
    for(int i=0;i<4;i++){
        omp_set_num_threads(n[i]);

        double start_time = omp_get_wtime();
        double t = 0.0, nrmu, nrmv;
        double u[N][N], v[N][N], du[N][N], dv[N][N];
        
        FILE *fptr = fopen("part1.dat", "w");
        fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");
        
        // initialize the state
        init(u, v);
        
        // time-loop
        for (int k=0; k < M; k++){
            // track the time
            t = dt*k;
            // evaluate the PDE
            dxdt(du, dv, u, v);
            // update the state variables u,v
            step(du, dv, u, v);
            if (k%m == 0){
                // calculate the norms
                nrmu = norm(u);
                nrmv = norm(v);
                printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
                fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
            }
        }
        
        fclose(fptr);
        double end_time = omp_get_wtime();
        total_times[i] = end_time - start_time;
        printf("number threads:%d, spent time: %f\n",n[i],total_times[i]);
    }
    
    FILE *fptr = fopen("part1_timeAnalysis.dat", "w");
    fprintf(fptr, "numberThread\t\ttime\t\tspeedup\t\teffciency\n");
    for(int i =0;i<4;i++){
        fprintf(fptr, "%d\t\t%f\t\t%f\t\t%f\n", n[i], total_times[i], total_times[0]/total_times[i],total_times[0]/total_times[i]/n[i]);
    }

	return 0;
}