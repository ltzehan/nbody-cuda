//
//	Manages control flow of the simulation
//	Will also manage OpenGL visualization when implemented
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simulation.h"
#include "vtkwriter.h"

#define BLOCK_SIZE 256
#define SOFTENING 1E-9f
#define TIME_STEP 0.01

Simulation::Simulation(const Config& config) : config(config) {
	
	// initialize particles
	pt = new Particles(config.n);

	this->start();

}

Simulation::~Simulation() {
	delete pt;
}

// control loop
void Simulation::start() {

	VTKWriter vtkw;
	
	for (int i = 0; i < config.frames; i++) {
		
		//update_particles();

		vtkw.write_pos(pt->d_pos, config.n);

	}

}

// advance simulation 
void Simulation::update_particles() {
	
	int grid_size = (config.n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	calc_interactions<<<grid_size, BLOCK_SIZE>>>(pt->d_pos, pt->d_vel, pt->d_acc, config.n);

}

// calculates change in acceleration due to gravity
__device__
void calc_acc(float4 pos_i, float4 pos_j, float4& acc) {

	float3 r = {
		pos_j.x - pos_i.x,
		pos_j.y - pos_i.y,
		pos_j.z - pos_i.z
	};
	
	// using softening parameter
	float dist = r.x * r.x + r.y * r.y * r.z * r.z + SOFTENING;
	float inv_distcb = 1.0f / sqrtf(dist * dist * dist);

	acc.x += r.x * inv_distcb;
	acc.y += r.y * inv_distcb;
	acc.z += r.z * inv_distcb;

}

__global__
void calc_interactions(float4* d_pos, float4* d_vel, float4* d_acc, const int n) {

	extern __shared__ float4 s_pos[];

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	float4 pos = d_pos[id];
	float4 acc = {0, 0, 0, 0};

	// loops through all particles by tiles
	int tile = 0;
	for (int i = 0; i < n; i += BLOCK_SIZE) {

		// id for thread in this tile
		int tid = tile * blockDim.x + threadIdx.x;

		// load particle positions into shared memory
		s_pos[threadIdx.x] = d_pos[tid];
		__syncthreads();

		// update current body's accleration with next tile of bodies
		for (int i = 0; i < blockDim.x; i++) {
			calc_acc(pos, s_pos[i], acc);
		}
		__syncthreads();

		tile++;

	}

	d_acc[id] = acc;

}
