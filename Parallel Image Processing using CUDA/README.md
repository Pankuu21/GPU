# 
# Parallel Image Processing using CUDA

## Overview
This project implements parallel image processing using CUDA. It performs two transformations on an input RGB image:

1. **Inverted Grayscale Transformation**: Converts an RGB image to grayscale and vertically inverts it.
2. **Thomas Transformation**: Applies the transformation \( \text{floor}(0.5 \times R) + \text{floor}(\sqrt{G}) + B \) on each pixel.

Both transformations are performed in parallel using CUDA kernels.

## File Structure
- `main.cu` - CUDA C++ implementation of image processing.
- `cuda.out` - Output file containing processed images.
- `cuda_timing.out` - Execution time recorded for CUDA processing.

## Requirements
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- g++ and nvcc compiler.

## Compilation & Execution
### 1. Compile the program using nvcc:
```sh
nvcc -o image_processing main.cu
```

### 2. Run the executable with an input file:
```sh
./image_processing input.txt
```

### 3. Output files:
- `cuda.out` will contain two transformed images in matrix format.
- `cuda_timing.out` will store the execution time in seconds.

## Implementation Details
- **Memory Allocation**: The program uses `cudaMalloc` for memory allocation and `cudaFree` for cleanup.
- **Kernel Execution**: The kernels are executed with a grid-stride loop ensuring parallel computation.
- **Performance Timing**: Measured using `std::chrono::high_resolution_clock`.

## Notes
- The input image should be in the format:
  ```
  M N  # Image dimensions (M rows, N columns)
  R G B values in row-major order
  ```
- The program automatically handles memory allocation and deallocation to optimize performance.

## Acknowledgments
This project is a part of the **CS6023: GPU Programming** course assignment at IIT Madras.