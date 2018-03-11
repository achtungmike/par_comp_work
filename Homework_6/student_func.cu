/*
Mike Lape
Homework 6 - Udacity Problem Set 3
4 Parts

	1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

#include "reference_calc.cpp"
#include "utils.h"



// Kernels below...
__global__ void scanHisto(unsigned int*  d_bins, unsigned int *result, int size)
{
    extern __shared__ int data[];
    
	// Who am I?
	int tid  = threadIdx.x;
    	
	// Where are we?
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    	

    // Load our part of histo into shared memory.
    if(myId < size)
    {
        data[tid] = d_bins[myId];
    }
	// Wait for everyone.
    __syncthreads();  
    
    // Hillis-Steele Scan
    for (int i = 1; i < size;i *= 2)
    {
        if (tid >= i)
        {
            data[tid] += data[tid - i];
        }
        __syncthreads();        
    }
		
    if(myId < size)
    {
        result[myId] = data[tid];
    }
}

__global__ void hist(const float* const d_in, unsigned int * d_out, const float range, const int min, const int numBins, int size)
{
	// Where are we?
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if( idx < size)
    {
		//Calculate the proper bin, and then add 1 to it in main device memory
        //bin = (lum[i] - lumMin) / lumRange * numBins
        int bin = ((d_in[idx] - min) / range) * numBins;
        atomicAdd(&(d_out[bin]), 1);
    }
	
	
	
}


__global__ void min_max_kernel(const float* const d_in, float* d_out, bool is_max=true)
{

	extern __shared__ float part[];

	// Locate where we are.
	int tid = threadIdx.x;
	int idx = blockIdx.x *  blockDim.x + tid;

	// load into shared memory
	part[tid] = d_in[idx];
	__syncthreads();
	
	for(unsigned int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if(tid < i){
			// if we are looking for max
			if(is_max)
			{
				part[tid] = max(part[tid], part[tid + i]);	
			}
			// else we are looking for the min
			else
			{
				part[tid] = min(part[tid], part[tid + i]);	
			}
		}
		
		// Sync everyone up.
		__syncthreads();
	}
	
	// Write your part back out to global memory
	if(tid == 0)
	{
		d_out[blockIdx.x] = part[tid];
	}
}

void min_max(const float* const d_in, float &min_logLum, float &max_logLum, 
				const size_t numRows, const size_t numCols)
{

	const int BLOCK_SIZE = numCols;
	const int GRID_SIZE  = numRows;
		
	// declare GPU memory pointers
	float * d_tmp;
	float * d_max;
	float * d_min;
		
	// allocate GPU memory
	cudaMalloc((void **) &d_tmp, GRID_SIZE*sizeof(float));
	cudaMalloc((void **) &d_max, sizeof(float));
	cudaMalloc((void **) &d_min, sizeof(float));


	checkCudaErrors(cudaMemset(d_tmp, 0, GRID_SIZE*sizeof(float)));
	

	// Find min
	// Have each block find it's min
	min_max_kernel<<<GRID_SIZE,BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_in, d_tmp, false);
	
	// Now use above to calculate global min
	min_max_kernel<<<1, GRID_SIZE, GRID_SIZE*sizeof(float)>>>(d_tmp, d_min, false);
	

	// Move results to host.
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));

	// Clean up
	checkCudaErrors(cudaFree(d_tmp));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_min));
	
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

/*  
1) Find min and max value in logLuminance
		Input: logLuminance 
		Output: min_logLum, max_logLum
  
2) Find range of values in logLuminance
  
3) Generate Histogram of all values in logLuminance
		Using the formula:
		bin = (lum[i] - lumMin) / lumRange * numBins
		lumRange: calculated in step 2
		lumMin: calculated in step 1
		numBins: handed in
		lum[i]: current value in luminance array being considered
	
4) Do exlusive scan on histogram to get cumulative distribution
		of luminance array, put this in d_cdf.
		
*/


	// Step 1: Find the min and max
	min_max(d_logLuminance, min_logLum, max_logLum, numRows, numCols);
  
  
	// Step 2: Find range:
	float lumRange = max_logLum - min_logLum;
  

	// Step 3: Generate histogram
	//	bin = (lum[i] - lumMin)/lumRange * numBins

	// Device bins ptr
	unsigned int  *d_bins;
	
	// Setup device memory
	checkCudaErrors(cudaMalloc((void **) &d_bins, numBins*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_bins,0,numBins*sizeof(unsigned int)));
	
	// Call kernel
	hist<<<numRows, numCols>>>(d_logLuminance, d_bins, lumRange, min_logLum, numBins, numRows*numCols);

    
	// Step 4:  Exclusive scan on histo to get cumulative distro

	scanHisto<<<1, numBins, numBins*sizeof(unsigned int)>>>(d_bins, d_cdf, numBins);
	checkCudaErrors(cudaFree(d_bins));

	
	
}
