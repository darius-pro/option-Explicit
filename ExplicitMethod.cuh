////////////////////////////////////////////////////////////
//
//	BSPDE	- holds upper/lower/terminal boundary conditions
//			- holds BS + Option parameters
//			- a,b,c,d = BSEq co-efficients
//			- t(i), x(j) = time, stock-price at i,j
//			- A,B,C,D = finite difference co-efficients
//
//	Solve	- Takes a 2D array and solves the Vector for
//			  time-step i-1#
//
//	v		- Extrapolates array to find price at non-int s0
//
////////////////////////////////////////////////////////////

#pragma once

#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>

class BSPDE
{
public:
	BSPDE(double s0_, double r_, double sigma_, double K_, double T_, double sl_, double su_, int imax, int jmax);
	__device__ double a(double t, double S);
	__device__ double b(double t, double S);
	__device__ double c(double t, double S);

	__device__ double t(double i);
	__device__ double S(int j);

	__device__ double f(int j);
	__device__ double fl(int i);
	__device__ double fu(int i);

	__host__ double f_h(int j);
	__host__ double fl_h(int i);
	__host__ double fu_h(int i);

	__device__ double A(int i, int j);
	__device__ double B(int i, int j);
	__device__ double C(int i, int j);

	double s0, r, sigma, K, T, sl, dt, dS;
};

__global__ void Solve(BSPDE *PDE_, double *V, int imax, int jmax);
__global__ void v(double *outputPtr, double t, double *V, BSPDE *PDE_, int jmax);