// Mike Lape
// Homework 5 - Udacity Problem Set 2
// 5 Parts
//      separateChannels                        X
//      gaussianBlur                            X
//      allocateMemoryAndCopyToGPU              X
//      cleanup                                 X
//      your_gaussian_blur ("main function")    X


//#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int2 pos_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
  
  // flatten pos into 1D
  const int pos_1d = (pos_2d.y * numCols) + pos_2d.x;


  // don't run off the end of the image
  if ( pos_2d.x >= numCols || pos_2d.y >= numRows )
  {
      return;
  }
  

    // augment clamping method shown in reference_calc
  float result = 0.0f;
      
  //For every value in the filter around the pixel filter_x and filter_y
  for (int filter_y = 0; filter_y < filterWidth; filter_y++) 
  {
    for (int filter_x = 0; filter_x < filterWidth; filter_x++) 
    {
       
      //Find the global image position for this filter position
      //clamp to boundary of the image
	    int image_x = (pos_2d.x + filter_x) - (filterWidth / 2);
	    int image_y = (pos_2d.y + filter_y) - (filterWidth / 2);
	   
	    //watch the edge!
	    // if you are below 0, max part set to 0
	    // if you are over the edge, min part set to numCol/numRows -1
	    image_x = min(max(image_x, 0), (numCols -1));
	    image_y = min(max(image_y, 0), (numRows -1));
	  
	    // pull image value and filter value for calculation
	    float image_value = static_cast<float>(inputChannel[(image_y * numCols) + image_x]);
      float filter_value = filter[(filter_y * filterWidth) + filter_x];

      result += image_value * filter_value;
    }
  }
 
  // dump our calculated value
  outputChannel[pos_1d] = (unsigned char) result;

}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // Get the 2D coords or this thread and stuff in var
  const int2 pos_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
  
  // flatten pos into 1D
  const int pos_1d = (pos_2d.y * numCols) + pos_2d.x;

  // don't run off the end of the image
  if ( pos_2d.x >= numCols || pos_2d.y >= numRows )
  {
      return;
  }
  

  
  // Sep out color channels and assign to correct params
  // Use our 1D index for each color channel array
  // use 1D index to pull from input image and use x, y, and z 
  // which rep r, g, b.
  redChannel[pos_1d]    =   inputImageRGBA[pos_1d].x;
  greenChannel[pos_1d]  =   inputImageRGBA[pos_1d].y;
  blueChannel[pos_1d]   =   inputImageRGBA[pos_1d].z;
  
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));


  // We use this twice so just assign it a var name
  const int b_size = sizeof(float) * filterWidth * filterWidth;

  // allocate filter memory
  // wrap in checkCudaErrors
  // Use cudaMalloc
  // give ptr to device memory
  // filterWidth is a box so area is filterWidth * filterWidth
  checkCudaErrors(cudaMalloc(&d_filter, b_size));

  
  // copy filter from host to device
  // wrap in checkCudaErrors
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, b_size, cudaMemcpyHostToDevice));
  

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  // Could probably go up to 1024 threads per block but for compat stay at or below 512
  // and I like a simple square. 22 * 22 * 1 = 484, largest square under 512
  const dim3 blockSize(22, 22, 1); 

  //remember the +1 to each dim
  const dim3 gridSize((numCols / blockSize.x) + 1, (numRows / blockSize.y) + 1, 1);

  //Call separateChannels kernel.
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

  
  // Call gausian kernel for each color.
  // RED
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

  // GREEN
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
  
  // BLUE
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
  
  
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
                                             d_outputImageRGBA, numRows, numCols);
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated

void cleanup() 
{
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
