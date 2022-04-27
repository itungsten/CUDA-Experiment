#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f
#define BLK 32

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

__global__ void bodyForce(Body *p, float dt, int n)
{
    // index of updated body
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // Resultant force on x,y,z axes
    float Fx=0, Fy=0, Fz=0;
    // iterate on all the bodies
    for (int j = 0; j < n; j++)
    {
        float dx = p[j].x - p[idx].x;
        float dy = p[j].y - p[idx].y;
        float dz = p[j].z - p[idx].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        // accumualte resultant force on x,y,z axes
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    // update velocity on x,y,z axes
    p[idx].vx += dt * Fx;
    p[idx].vy += dt * Fy;
    p[idx].vz += dt * Fz;
}

__global__ void integrate_position(Body *p, float dt, int n)
{
    // index of updated body
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // update position on x,y,z axes
    p[idx].x += p[idx].vx * dt;
    p[idx].y += p[idx].vy * dt;
    p[idx].z += p[idx].vz * dt;
}

int main(const int argc, const char **argv)
{

    /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

    int nBodies = 2 << 11;
    int salt = 0;
    if (argc > 1)
        nBodies = 2 << atoi(argv[1]);

    /*
   * This salt is for assessment reasons. Tampering with it will result in automatic failure.
   */

    if (argc > 2)
        salt = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf; cudaMallocManaged(&buf, bytes); Body *p = (Body *)buf;
    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    size_t blockNum = (nBodies + BLK - 1) / BLK;
    double totalTime = 0.0;

    /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

    /*******************************************************************/
    // Do not modify these 2 lines of code.
    for (int iter = 0; iter < nIters; iter++)
    {
        StartTimer();
    /*******************************************************************/
        bodyForce<<<blockNum, BLK>>>(p, dt, nBodies); // compute forces
        integrate_position<<<blockNum, BLK>>>(p,dt,nBodies); // update positions
        if(iter == nIters-1)cudaDeviceSynchronize(); //sync memory
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
    cudaFree(buf);
}
