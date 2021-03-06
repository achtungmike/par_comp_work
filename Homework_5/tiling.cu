#include <stdio.h>
#include "gputimer.h"
//#include "utils.h"

const int N= 1024;	// matrix size will be NxN
const int K = 32;	// Our tile size

int compare_matrices(float *gpu, float *ref, int N)
{
        int result = 0;
        for(int j=0; j < N; j++)
        {
		for(int i=0; i < N; i++)
		{
                	if (ref[i + j*N] != gpu[i + j*N])
                   		{result = 1;}
		//	printf("%d\t", (int)gpu[i + j*N]);
		}
	//printf("\n");
	}
 return result;
	
}


// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat, int N)
{
        for(int j=0; j < N * N; j++)
                mat[j] = (float) j;
}

// The following functions and kernels are for your references
void 
transpose_CPU(float in[], float out[])
{
	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
      		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ void 
transpose_serial(float in[], float out[])
{
	for(int j=0; j < N; j++)
		for(int i=0; i < N; i++)
			out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per row of output matrix
__global__ void 
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	for(int j=0; j < N; j++)
		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}



// Write two tiled versions of transpose -- One using shared memory. 
// To be launched with one thread per element, in KxK threadblocks.
// You will determine for each thread (x,y) in tile the element (i,j) of global output matrix. 

__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{

	int x = blockIdx.x * K + threadIdx.x;
  	int y = blockIdx.y * K + threadIdx.y;

	// define our input index for this thread and output idex for this thread.	
	int in_idx = x + N * y;
	int out_idx = y + N * x;

	for(int i = 0; i < K; i+= N)
	{
		out[out_idx+i] = in[in_idx +i * N];
	}


		
}

__global__ void 
transpose_parallel_per_element_tiled_shared(float in[], float out[])
{
	// Same algo as above but using shared memory.

	// Define our tile using K	
        __shared__ float tile[K][K+1];
	
	int idx = threadIdx.x;
	
	int x  = blockIdx.x * K;
	int y  = blockIdx.y * K;
	
	// Read tile into shared memory	
	for ( int i=0; i < K; i++)
	{
		tile[idx][i] = in[( y + i ) * N + ( x + idx ) ];
	}

	// sync the threads
	__syncthreads();
	
	// write the data out of shared memory into global memory
	for ( int i=0; i < K; i++)
	{
		out [( x + i )* N + ( y + idx) ] = tile[i][idx];
	}

}

int main(int argc, char **argv)
{
	int numbytes = N * N * sizeof(float);

	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);

	fill_matrix(in, N);
	transpose_CPU(in, gold);

	float *d_in, *d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;


    timer.Start();
	transpose_serial<<<1,1>>>(d_in, d_out);
	timer.Stop();
    for (int i=0; i < N*N; ++i){out[i] = 0.0;}
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_serial: %g ms.\nVerifying ...%s\n", 
		   timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

   
    cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice); //clean d_out
    timer.Start();
	transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
	timer.Stop();
    for (int i=0; i < N*N; ++i){out[i] = 0.0;}  //clean out
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms.\nVerifying ...%s\n", 
		    timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

    cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice); //clean d_out
    // Tiled versions
    //const int K= 16;
    dim3 blocks_tiled(N/K,N/K);
	dim3 threads_tiled(K,K);
	timer.Start();
	transpose_parallel_per_element_tiled<<<blocks_tiled,threads_tiled>>>(d_in, d_out);
	timer.Stop();
    for (int i=0; i < N*N; ++i){out[i] = 0.0;}
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

    cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice); //clean d_out
    dim3 blocks_tiled_sh(N/K,N/K);
	dim3 threads_tiled_sh(K,K);
     timer.Start();
	transpose_parallel_per_element_tiled_shared<<<blocks_tiled_sh,threads_tiled_sh>>>(d_in, d_out);
	timer.Stop();
    for (int i=0; i < N*N; ++i){out[i] = 0.0;}
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled_shared %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

	cudaFree(d_in);
	cudaFree(d_out);
}
