%%writefile test1.cu

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;
typedef long long ll;


#define tile_h 8
#define tile_w 8

using std::cin;
using std::cout;




_global_ void dkernel_con(long int *matrix, long int *filter, long int *result, int k, int s, int r, int h, int w, int c,int share_h,int share_w)
{
    int g_r,g_c;
    long int value;
    int pad_r = r / 2;
    int pad_s = s / 2;

    extern _shared_ long int sh_mem[];

    int num_tile_r=ceil((float)h/(float)tile_h);
    int f = blockIdx.y / num_tile_r;   
    //int tileRow = blockIdx.y % num_tile_r; 

    int output_cord_i = (blockIdx.y % num_tile_r) * tile_h + threadIdx.y;       
    int output_cord_j = blockIdx.x * tile_w + threadIdx.x;  

    int base_r = (output_cord_i - threadIdx.y) - pad_r;
    int base_c = (output_cord_j - threadIdx.x) - pad_s;
    long int sum = 0;
    for (int ch = 0; ch < c; ch++) {
        for (int curr_x = threadIdx.x; curr_x < (tile_w + s - 1); curr_x += blockDim.x) {
            for (int curr_y = threadIdx.y; curr_y < (tile_h + r - 1); curr_y += blockDim.y) {
                g_c = base_c + curr_x;
                value = 0;
                g_r = base_r + curr_y;  
                   
                int matrix_indx=g_c+(ch * h + g_r) * w;
                
                if (g_r < h &&  g_c < w &&g_c >= 0 &&g_r >= 0) {
                    
                    value = matrix[matrix_indx];
                }
                
                int sm_Indx = ch * ((share_h) * (share_w)) + curr_y * (share_w) + curr_x;
                sh_mem[sm_Indx] = value;
            }
        }
    }
    __syncthreads();

    
if (output_cord_i < h && output_cord_j < w && f < k) {

    for (int filter_r = r - 1; filter_r >= 0; filter_r--) {

        for (int ch = c - 1; ch >= 0; ch--) {
           
            for (int filter_s = s - 1; filter_s >= 0; filter_s--) {
                int sh_y = threadIdx.y + filter_r;  
                int sh_x = threadIdx.x + filter_s;  

                long int imgVal = sh_mem[ch * (share_h * share_w) + sh_y * share_w + sh_x];
                
                long int filter_Val = filter[f * (r * s * c) + ch * (r * s) + filter_r * s + filter_s];
                sum += imgVal * filter_Val;
            }
        }
    }
    
   result[(f * h + output_cord_i) * w + output_cord_j] = sum;




}




    






           
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /*Start Here/
    //allocation
    long int *g_mat,*g_result,*g_filter;
    cudaMalloc(&g_filter, sizeof(long int) * ((r c) s) * k);
    cudaMalloc(&g_result, sizeof(long int) ( h * w ) k);
    cudaMalloc(&g_mat,sizeof(long int)*((h*c) * w) );
    
    //mem copy
    cudaMemcpy(g_filter, h_filter, sizeof(long int)*  ((r c) s) * k, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat, h_mat, sizeof(long int) * ((h*c) * w) , cudaMemcpyHostToDevice);
    // calculating dimensions
    int grid_y=(ceil(float(h)/float(tile_h)))*k;
    int grid_x=(ceil(float(w)/float(tile_w)));

   int share_w  = tile_w + s - 1;   
    int share_h = tile_h + r - 1;

    dim3 block_dim(tile_w,tile_h);
    dim3 grid_dim(grid_x, grid_y);
    size_t share_size= sizeof(long int)*(c * share_h * share_w);

    dkernel_con<<<grid_dim,block_dim,share_size >>>(g_mat, g_filter, g_result, k, s, r, h, w, c,share_h,share_w);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_ans, g_result, sizeof(long int) ( h * w ) k, cudaMemcpyDeviceToHost);


    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    /$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}