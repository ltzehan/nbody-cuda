#pragma once

#include "float4.h"

// used for organizing property arrays of particles and not meant to be used as a function parameter
// vectors are stored in float4 which should be able to exploit speedup by coalesced access
struct Particles {

	// particle position
	float4* d_pos;
	// particle velocity
	float4* d_vel;
	// particle acceleration
	float4* d_acc;

	const int n;

	Particles(const int n);
	~Particles();

private:

	void init(float4* h_pos, float4* h_vel, float4* h_acc);

	void copy(float4* h_pos, float4* h_vel, float4* h_acc);

};
