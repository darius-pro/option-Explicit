#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "ExplicitMethod.cuh"
#include <iostream>
#include <Windows.h>


using namespace std;


int main()
{
	double s0_ = 100.0, r_ = 0.05, sigma_ = 0.2;
	double K_ = 100.0, T_ = 1./12, sl_ = 0, su_ = 2.0*s0_;
	const int imax = 3000, jmax = 500;

	BSPDE PDE(s0_, r_, sigma_, K_, T_, sl_, su_, imax, jmax);

	double V[jmax + 1];

	V[0] = PDE.fl_h(imax);
	V[jmax] = PDE.fu_h(imax);
	for (int j = 1; j < jmax; j++)
	{
		V[j] = PDE.f_h(j);
	}

	double *d_V;
	BSPDE *d_PDE;

	cudaMalloc(&d_PDE, sizeof(PDE));
	cudaMemcpy(d_PDE, &PDE, sizeof(PDE), cudaMemcpyHostToDevice);

	cudaMalloc(&d_V, sizeof(V));
	cudaMemcpy(d_V, &V, sizeof(V), cudaMemcpyHostToDevice);

	Solve << <1, jmax - 1 >> > (d_PDE, d_V, imax, jmax);


	cudaMemcpy(&V, d_V, sizeof(V), cudaMemcpyDeviceToHost);

	double v0;
	double *d_v0;
	cudaMalloc(&d_v0, sizeof(double));

	v << <1, 1 >> >(d_v0, 0, d_V, d_PDE, jmax);

	cudaMemcpy(&v0, d_v0, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&V, d_V, sizeof(double)*(jmax + 1), cudaMemcpyDeviceToHost);

	cudaFree(d_v0);
	cudaFree(d_V);
	cudaFree(d_PDE);

	cout << v0 << endl;

	system("pause");
	return 0;
}