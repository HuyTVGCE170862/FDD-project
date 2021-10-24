#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#define max(x,y) ((x>y)? x:y)
#define min(x,y) ((x<y)? x:y)
#define np 32
#define nfiles 32*1
#define filesize 10240
#define ck 32
#define range 256
float precision = 0.0001;
int main (int argv, char ** argc) {
    int MAXITER = 300;
    double start = 0,end = 0, total_time = 0;

    MPI_Init(&argv,&argc);
    int node,csize,i,temp;
    MPI_Comm_rank(MPI_COMM_WORLD,&node);
    MPI_Comm_size(MPI_COMM_WORLD,&csize);

    populate_data(node); //read from files and populate data
    populate_clusters(cK/np,node); //cK/np is the number of fucking clusters per node

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    initialize_all_means(cK/np);

    int iter = 0;
    while((iter<MAXITER) && (!check_stop_condition(cK/np))){

        copy
    }

}