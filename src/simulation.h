#pragma once

#include "config.h"
#include "particles.h"

struct Simulation {

	Simulation(const Config& config);
	~Simulation();

	void start();
	
private:

	const Config config;
	Particles* pt;

	void update_particles();

};

__global__
void calc_interactions(float4* d_pos, float4* d_vel, float4* d_acc, const int n);
