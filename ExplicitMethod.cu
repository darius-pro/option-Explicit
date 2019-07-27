#include "ExplicitMethod.cuh"
#include <device_functions.h>
#include <cmath>
#include <stdio.h>

BSPDE::BSPDE(double s0_, double r_, double sigma_, double K_, double T_, double sl_, double su_, int imax, int jmax) : s0(s0_), r(r_), sigma(sigma_), K(K_), T(T_), sl(sl_)
{
	dt = T / imax;
	dS = (su_ - sl_) / jmax;
}

__device__ double BSPDE::a(double t, double S)
{
	return -0.5*sigma*sigma*S*S;
}

__device__ double BSPDE::b(double t, double S)
{
	return -r*S;
}

__device__ double BSPDE::c(double t, double S)
{
	return r;
}

__device__ double BSPDE::t(double i)
{
	return i*dt;
}

__device__ double BSPDE::S(int j)
{
	return sl + j*dS;
}

__device__ double BSPDE::f(int j)
{
	if (K > S(j))
		return K - S(j);
	else
		return 0.0;
}

__device__ double BSPDE::fl(int i)
{
	return K*exp(-r*(T - dt*i));
}

__device__ double BSPDE::fu(int i)
{
	return 0.0;
}

__device__ double BSPDE::A(int i, int j)
{
	return (dt / dS) * (b(t(i), S(j)) / 2.0 - a(t(i), S(j)) / dS);
}

__device__ double BSPDE::B(int i, int j)
{
	return 1.0 - dt*c(t(i), S(j)) + 2.0 * dt*a(t(i), S(j)) / (dS*dS);
}

__device__ double BSPDE::C(int i, int j)
{
	return -(dt/dS)*(b(t(i), S(j)) / 2.0 + a(t(i), S(j)) / dS);
}


__host__ double BSPDE::f_h(int j)
{
	if (K > (sl + j*dS))
		return K - (sl + j*dS);
	else
		return 0.0;
}

__host__ double BSPDE::fl_h(int i)
{
	return K*exp(-r*(T - dt*i));
}

__host__ double BSPDE::fu_h(int i)
{
	return 0.0;
}

__global__ void Solve(BSPDE *PDE_, double *V, const int imax, const int jmax)
{
	int index = threadIdx.x + 1;
	__shared__ double V_[1026];
	__shared__ double V_minus[1026];
	V_[index] = V[index];
	__syncthreads();
	for (int i = imax; i >= 1; i--)
	{
		__syncthreads();
		if (threadIdx.x == 0)
		{
			V_minus[0] = PDE_->fl(i - 1);
			V_minus[jmax] = PDE_->fu(i - 1);
		}
		V_minus[index] = PDE_->A(i, index)*V_[index - 1] + PDE_->B(i, index) * V_[index]
			+ PDE_->C(i, index)*V_[index + 1];
		__syncthreads();
		V_[index] = V_minus[index];
		__syncthreads();
	}
	V[index] = V_[index];
}

__global__ void v(double *outputPtr, double t, double *V, BSPDE *PDE_, int jmax)
{
	int j_lower = ((PDE_->s0 - PDE_->sl) / PDE_->dS);
	int j_upper = j_lower + 1;
	double PCT = (PDE_->s0 - PDE_->S(j_lower)) / (PDE_->S(j_upper) - PDE_->S(j_lower));
	*outputPtr = V[j_lower] + PCT*(V[j_upper] - V[j_lower]);
}