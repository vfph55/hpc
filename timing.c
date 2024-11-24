#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double time_ij(int N){

	int i, j;
	
	double A[N][N];
	double B[N][N];
	double C[N][N];
	
	clock_t t0 = clock();
	for(i=0; i<N; i++){
		for (j=0; j<N; j++){
			C[i][j] = A[i][j] + B[i][j];
		}
	}
	double t1 = ((double)clock() - t0) / CLOCKS_PER_SEC;
	return t1;
}

double time_ji(int N){

	int i, j;
	
	double A[N][N];
	double B[N][N];
	double C[N][N];
	
	clock_t t0 = clock();
	for(j=0; i<N; i++){
		for (i=0; j<N; j++){
			C[i][j] = A[i][j] + B[i][j];
		}
	}
	double t1 = ((double)clock() - t0) / CLOCKS_PER_SEC;
	return t1;
}

int main(int argc, char **argv){

	int N = atoi(argv[1]);
	int M = atoi(argv[2]);
	double t;
	
	printf("3D version:\n");
	
	/*
		Your 3D code here
	*/
	
	printf("2D version:\n");
	
	t=0.0;
	for (int m=0; m < M; m++){
		t += time_ij(M);
	}
	printf("\tij: %2.16f s\n",t/M);
	
	t=0.0;
	for (int m=0; m < M; m++){
		t += time_ij(M);
	}
	printf("\tji: %2.16f s\n",t/M);

	
	return 0;
}