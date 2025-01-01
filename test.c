#include <omp.h>
#include <stdlib.h>
#include <stdio.h>


int main(){

    #pragma omp parallel
    {
        int id = omp_get_thread_num();  // 获取线程 ID
        printf("hello(%d)\n",id);
    }
    

    return 0;
}