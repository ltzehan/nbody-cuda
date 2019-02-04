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

__device__
float4 calc_acc(const float4 pos_i, const float4 pos_j, float4 acc);

__device__
float4 update_acc(const float4* d_pos, const int n);

__global__
void update_kernel(float4* d_pos, float4* d_vel, float4* d_acc, const int n);