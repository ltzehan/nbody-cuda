#pragma once

#include "float3.h"

// used for organizing property arrays of particles and not meant to be used as a method parameter
// vectors are stored in float3 which should be able to exploit speedup by coalesced access
struct Particles {

	// particle position
	float3* d_pos;
	// particle velocity
	float3* d_vel;
	// particle accleration
	float3* d_acc;

	const int n;

	Particles(const int n);
	~Particles();

private:

	void init();

};