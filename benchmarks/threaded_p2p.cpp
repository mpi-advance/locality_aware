#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>
#include <omp.h>

void p2p(double* sendbuf, double* recvbuf, int n, int s, int proc,
		MPI_Request* send_req, MPI_Request* recv_req, int tag)
{
    for (int i = 0; i < n; i++)
    {
        MPI_Isend(&(sendbuf[i*s]), s, MPI_DOUBLE, proc, tag, MPI_COMM_WORLD, &(send_req[i]));
	MPI_Irecv(&(recvbuf[i*s]), s, MPI_DOUBLE, proc, tag, MPI_COMM_WORLD, &(recv_req[i]));
    }
    MPI_Waitall(n, send_req, MPI_STATUSES_IGNORE);
    MPI_Waitall(n, recv_req, MPI_STATUSES_IGNORE);
}

void threaded_p2p(double* sendbuf, double* recvbuf, int n, int s, int proc,
		MPI_Request* send_req, MPI_Request* recv_req, int tag)
{
#pragma omp parallel shared(sendbuf, recvbuf, send_req, recv_req)
{
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    int thread_n = n / num_threads;
    int first_n = thread_n * thread_id;
    int extra = n % num_threads;
    if (extra > thread_id)
    {
        thread_n++;
        first_n += thread_id;
    }
    else first_n += extra;

    p2p(&(sendbuf[first_n*s]), &(recvbuf[first_n*s]), thread_n, s, proc, 
		    &(send_req[first_n]), &(recv_req[first_n]), tag+thread_id);
}

}

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) printf("Provided %d of (%d, %d, %d, %d)\n", provided,
		    MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    int ppn, local_rank;
    MPI_Comm_size(xcomm->local_comm, &ppn);
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    gpuSetDevice(local_rank);
    if (rank == 0) printf("PPN %d\n", ppn);

    int nthreads = 2;
    omp_set_num_threads(nthreads);
    if (rank == 0) printf("Set Num Threads : %d\n", nthreads);

    int proc = rank + ppn;
    if (rank / ppn) proc = rank - ppn;

    int max_i = 10;
    int max_s = pow(2, max_i);
    int max_j = 10;
    int max_n = pow(2, max_j);
    int max_n_iter = 100;
    double t0, tfinal;
    srand(time(NULL));

    int max_buf = max_s * max_n;
    std::vector<double> send_data(max_buf);
    std::vector<double> recv_data(max_buf);
    std::vector<double> recv_gpu_data(max_buf);
    std::vector<double> recv_cpu_data(max_buf);
    std::vector<double> recv_thread_data(max_buf);
    for (int j = 0; j < max_buf; j++)
        send_data[j] = rand();
    std::vector<MPI_Request> send_req(max_n);
    std::vector<MPI_Request> recv_req(max_n);

    double* send_data_d;
    double* recv_data_d;
    gpuMalloc((void**)(&send_data_d), max_buf*sizeof(double));
    gpuMalloc((void**)(&recv_data_d), max_buf*sizeof(double));
    gpuMemcpy(send_data_d, send_data.data(), max_buf*sizeof(double), gpuMemcpyHostToDevice);

    for (int i = 0; i < max_i; i++)
    {
        int s = pow(2, i);
        for (int j = 0; j < max_j; j++)
	{
            int n = pow(2, j);
	    if (rank == 0) printf("Testing %d Msgs, Each of Size %d Doubles\n", n, s);

            int n_iter = max_n_iter;
	    if (s > 4096)
		n_iter /= 10;
	    if (n > 100)
		n_iter /= 10;
	    if (n_iter < 1) n_iter = 1;

	    // Standard GPUDirect P2P
	    p2p(send_data_d, recv_data_d, n, s, proc, 
			    send_req.data(), recv_req.data(), 0);
	    gpuMemcpy(recv_gpu_data.data(), recv_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);

	    // Copy to CPU P2P
	    gpuMemcpy(send_data.data(), send_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
	    p2p(send_data.data(), recv_data.data(), n, s, proc,
			    send_req.data(), recv_req.data(), 0);
	    gpuMemcpy(recv_data_d, recv_data.data(), n*s*sizeof(double), gpuMemcpyHostToDevice);
	    gpuMemcpy(recv_cpu_data.data(), recv_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
	    for (int i = 0; i < n*s; i++)
	    {
		if (fabs(recv_gpu_data[i] - recv_cpu_data[i]) > 1e-6)
		{
			printf("C2CPU[%d] different! %e vs %e\n", i, recv_gpu_data[i], recv_cpu_data[i]);
			break;
		}
	    }

	    // Copy to Threaded CPU P2P
	    gpuMemcpy(send_data.data(), send_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
	    threaded_p2p(send_data.data(), recv_data.data(), n, s, proc,
			    send_req.data(), recv_req.data(), 0);
	    gpuMemcpy(recv_data_d, recv_data.data(), n*s*sizeof(double), gpuMemcpyHostToDevice);
	    gpuMemcpy(recv_thread_data.data(), recv_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
	    for (int i = 0; i < n*s; i++)
	    {
		if (fabs(recv_gpu_data[i] - recv_thread_data[i]) > 1e-6)
		{
			printf("Threaded[%d] different! %e vs %e\n", i, recv_gpu_data[i], recv_thread_data[i]);
			break;
		}

	    }


	    // Time GPUDirect
	    p2p(send_data_d, recv_data_d, n, s, proc,
                            send_req.data(), recv_req.data(), 0);
	    gpuDeviceSynchronize();
	    MPI_Barrier(MPI_COMM_WORLD);
	    t0 = MPI_Wtime();
	    for (int i = 0; i < n_iter; i++)
	    {
	        p2p(send_data_d, recv_data_d, n, s, proc,
                            send_req.data(), recv_req.data(), 0);
	    }
	    tfinal = (MPI_Wtime() - t0) / n_iter;
	    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	    if (rank == 0) printf("GPUDirect %e\n", t0);
		    

	    // Time Copy to CPU
            gpuMemcpy(send_data.data(), send_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
            p2p(send_data.data(), recv_data.data(), n, s, proc,
                            send_req.data(), recv_req.data(), 0);
            gpuMemcpy(recv_data_d, recv_data.data(), n*s*sizeof(double), gpuMemcpyHostToDevice);
	    gpuDeviceSynchronize();
	    MPI_Barrier(MPI_COMM_WORLD);
	    t0 = MPI_Wtime();
	    for (int i = 0; i < n_iter; i++)
	    {
                gpuMemcpy(send_data.data(), send_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
                p2p(send_data.data(), recv_data.data(), n, s, proc,
                            send_req.data(), recv_req.data(), 0);
                gpuMemcpy(recv_data_d, recv_data.data(), n*s*sizeof(double), gpuMemcpyHostToDevice);
            }
	    tfinal = (MPI_Wtime() - t0) / n_iter;
	    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	    if (rank == 0) printf("Copy to CPU %e\n", t0);


            // Copy to Threaded CPU P2P
            gpuMemcpy(send_data.data(), send_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
            threaded_p2p(send_data.data(), recv_data.data(), n, s, proc,
                            send_req.data(), recv_req.data(), 0);
            gpuMemcpy(recv_data_d, recv_data.data(), n*s*sizeof(double), gpuMemcpyHostToDevice);
	    gpuDeviceSynchronize();
	    MPI_Barrier(MPI_COMM_WORLD);
	    t0 = MPI_Wtime();
	    for (int i = 0; i < n_iter; i++)
	    {
                gpuMemcpy(send_data.data(), send_data_d, n*s*sizeof(double), gpuMemcpyDeviceToHost);
                threaded_p2p(send_data.data(), recv_data.data(), n, s, proc,
                            send_req.data(), recv_req.data(), 0);
                gpuMemcpy(recv_data_d, recv_data.data(), n*s*sizeof(double), gpuMemcpyHostToDevice);
            }
	    tfinal = (MPI_Wtime() - t0) / n_iter;
	    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	    if (rank == 0) printf("Threaded %e\n", t0);
	}
    }

    gpuFree(send_data_d);
    gpuFree(recv_data_d);

    MPIX_Comm_free(xcomm);

    MPI_Finalize();
    return 0;
}
