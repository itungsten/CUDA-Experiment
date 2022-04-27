#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define BLK 64
#define STRIDE 32

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void bodyForce(Body *p, float dt, int n)
{

    // index of updated body
    int idx = threadIdx.x + (blockIdx.x / STRIDE) * blockDim.x; 
    Body pi = p[idx];
    int startID = blockIdx.x % STRIDE;
    int blockNum = n / BLK;
    // shared_memory as caches 
    __shared__ float3 caches[BLK];
    float dx, dy, dz, distSqr, invDist, invDist3;
    // Resultant force on x,y,z axes
    float Fx=0, Fy=0, Fz=0;
    // iterate on bodies
    for (int currID = startID; currID < blockNum; currID += STRIDE)
    {
        Body tmp = p[currID * BLK + threadIdx.x];
        caches[threadIdx.x] = make_float3(tmp.x, tmp.y, tmp.z);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BLK; j++)
        {
            dx = caches[j].x - pi.x;
            dy = caches[j].y - pi.y;
            dz = caches[j].z - pi.z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = rsqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;
            // accumualte resultant force on x,y,z axes
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }
    // update velocity on x,y,z axes 
    // concurrently with atomic ops
    atomicAdd(&p[idx].vx, dt * Fx);
    atomicAdd(&p[idx].vy, dt * Fy);
    atomicAdd(&p[idx].vz, dt * Fz);
}

__global__ void integrate_position(Body *p, float dt, int n)
{
    // index of updated body
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // update position on x,y,z axes
    p[idx].x += p[idx].vx * dt;
    p[idx].y += p[idx].vy * dt;
    p[idx].z += p[idx].vz * dt;
}

int main(const int argc, const char **argv)
{

    int nBodies = 2 << 11;
    int salt = 0;
    if (argc > 1)
        nBodies = 2 << atoi(argv[1]);

    /*
   * This salt is for assessment reasons. Tampering with it will result in automatic failure.
   */

    if (argc > 2) salt = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf; cudaMallocHost(&buf, bytes);

    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    double totalTime = 0.0;

    size_t blockNum = (nBodies + BLK - 1) / BLK;

    float *bufDev; cudaMalloc(&bufDev, bytes); Body *pDev = (Body *)bufDev;
    /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

    cudaMemcpy(bufDev, buf, bytes, cudaMemcpyHostToDevice);
    /*******************************************************************/
    // Do not modify these 2 lines of code.
    for (int iter = 0; iter < nIters; iter++)
    {
        StartTimer();
    /*******************************************************************/
        bodyForce<<<blockNum * STRIDE, BLK>>>(pDev, dt, nBodies); // compute interbody forces
        integrate_position<<<blockNum, BLK>>>(pDev, dt, nBodies);
        if (iter == nIters - 1)cudaMemcpy(buf, bufDev, bytes, cudaMemcpyDeviceToHost);
    /*******************************************************************/
    // Do not modify the code in this section.
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
    checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
    checkAccuracy(buf, nBodies);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
    salt += 1;
#endif
    /*******************************************************************/

    /*
   * Feel free to modify code below.
   */
    cudaFree(bufDev);
    cudaFreeHost(buf);
}