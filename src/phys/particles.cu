#define _USE_MATH_DEFINES
#include <cmath>

// for portability's sake
#ifndef _MATH_DEFINES_DEFINED
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#endif

#include "particles.h"
#include "debug.h"
#include <random>

// Plummer model scale
const float SCALE = 1.0;

Particles::Particles(const int n) : n(n) {

	// allocate memory to device side variables
	gpu_check(cudaMalloc((void**)&d_pos, sizeof(float4) * n));
	gpu_check(cudaMalloc((void**)&d_vel, sizeof(float4) * n));
	gpu_check(cudaMalloc((void**)&d_acc, sizeof(float4) * n));

	// temporary host arrays for initialization 
	float4* h_pos = new float4[n];
	float4* h_vel = new float4[n];
	float4* h_acc = new float4[n];

	this->init(h_pos, h_vel, h_acc);
	this->copy(h_pos, h_vel, h_acc);

	delete[] h_pos;
	delete[] h_vel;
	delete[] h_acc;

}

Particles::~Particles() {

	gpu_check(cudaFree(d_pos));
	gpu_check(cudaFree(d_vel));
	gpu_check(cudaFree(d_acc));

}

// copy particle properties from host to device
void Particles::copy(float4* h_pos, float4* h_vel, float4* h_acc) {

	gpu_check(cudaMemcpy(d_pos, h_pos, sizeof(float4) * n, cudaMemcpyHostToDevice));
	gpu_check(cudaMemcpy(d_vel, h_vel, sizeof(float4) * n, cudaMemcpyHostToDevice));
	gpu_check(cudaMemcpy(d_acc, h_acc, sizeof(float4) * n, cudaMemcpyHostToDevice));

}

// initialize particle positions using Plummer model
// will include other cluster models in the future
void Particles::init(float4* h_pos, float4* h_vel, float4* h_acc) {

	using dstr = std::uniform_real_distribution<float>;

	dstr azimuth(0, 2 * M_PI);
	dstr elevation(-1, 1);
	dstr mass_ratio(0, 1);
	dstr velq(0, 1);
	dstr velg(0, 0.1);

	// initialize random number generator
	std::random_device rd;
	std::mt19937 rng(rd());

	for (int i = 0; i < n; i++) {

		// radial distance
		float r = SCALE / sqrt(pow(mass_ratio(rng), -0.666666) - 1);
		// spherical coordinates
		float azi = azimuth(rng);
		float elv = acos(elevation(rng)) - M_PI_2;

		// populate position host vectors
		h_pos[i] = make_float4(
			r * cos(elv) * cos(azi),
			r * cos(elv) * sin(azi),
			r * sin(elv)
		);

		// rejection sampling for velocities
		float q;
		float g;
		do {

			q = velq(rng);
			g = velg(rng);

		} while (g > pow((q*q)*(1 - (q*q)), 3.5));

		// velocity magntitude
		float v = q * M_SQRT2 * pow(1.0 + r * r, -0.25);
		// spherical coordinates
		azi = azimuth(rng);
		elv = acos(elevation(rng)) - M_PI_2;

		// populate velocity host vectors
		h_vel[i] = make_float4(
			v * cos(elv) * cos(azi),
			v * cos(elv) * sin(azi),
			v * sin(elv)
		);

		// populate acceleration host vectors
		h_acc[i] = make_float4(0, 0, 0);

	}

}