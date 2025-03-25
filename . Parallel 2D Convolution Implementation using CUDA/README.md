# Parallel 2D Convolution using CUDA

## Overview
This project implements parallel 2D convolution on multi-channel images using CUDA.  
It leverages shared memory and tiling to efficiently compute the convolution across multiple filters and channels.

## File Structure
- `convolution.cu` — CUDA C++ implementation for parallel convolution.
- `cuda.out` — Output file containing convolved results.
- `cuda_timing.out` — Execution time recorded for CUDA processing.

## Requirements
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- g++ and nvcc compiler.

## Compilation & Execution
### 1. Compile the program using nvcc:
```bash
nvcc -o convolution convolution.cu
2. Run the executable with an input file:
bash
Copy
Edit
./convolution < input.txt
3. Output files:
cuda.out will contain the convolved output in matrix form.

cuda_timing.out will store the execution time in seconds.

Input Format
First line: h w c (image height, width, channels)

Followed by h * w * c pixel values.

Next line: cf r s k (cf ignored, r and s filter dimensions, k number of filters)

Followed by r * s * c * k filter values.

Implementation Highlights
Tiled convolution with shared memory for faster computation.

Parallel execution with careful boundary handling.

Memory allocation and synchronization done using CUDA APIs.

Acknowledgments
This project is part of the CS6023: GPU Programming course assignment at IIT Madras.