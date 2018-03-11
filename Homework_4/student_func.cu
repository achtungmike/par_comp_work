// Mike Lape
// CS6068
// Homework 4
// Color to Greyscale Conversion


#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  // Map the 2D block 
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  
  if ( (x < numRows) && (y < numCols)) {
  
    // Our thread is inside the picture so convert to 1D index for color changing
    int idx = ( x * numCols) + y;
    
    // Pull out the color pixel.
    uchar4 colPixel = rgbaImage[idx];

    // Convert the pixel to greyscale and shove it in the greyImage
    unsigned char greyPixel = (unsigned char) ((0.299 * colPixel.x) + (.587 * colPixel.y) + (.114 * colPixel.z));
    greyImage[idx]  = greyPixel;
  
  }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  // Make sure our block size is less than 512 (max number of blocks for older Nvidia cards)
  //floor(sqrt(512)) = 22 so 22 blocks in x and y dir, leaving out z
  // Put this in a constant so we could easily update it if using with a 1024 block compatible card.
  const int BLOCK = 22;
  const dim3 blockSize(BLOCK, BLOCK, 1);
  
  // Split up blocks into grids taking into account size of image
  int blockRow = ( numRows / blockSize.x ) + 1;
  int blockCol = ( numCols / blockSize.y ) + 1;
  
  //shape our grid
  const dim3 gridSize( blockRow, blockCol, 1); 
  
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
