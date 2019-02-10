//
//	Manages control flow of the simulation
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "debug.h"
#include "simulation.h"

#define BLOCK_SIZE 256
#define SOFTENING 1E-9f
#define TIME_STEP 0.01

Simulation::Simulation(const Config& config) : config(config) {
	
	// initialize particles
	pt = new Particles(config.n);

	// initialize display handler
	glhandler = new GLHandler(this);

	// initialize VTK output writer if necessary
	vtkwriter = new VTKWriter();
	
	// initialize host side array for positions
	vtk_pos = new float4[config.n];

	// start simulation through display handler
	glhandler->loop();

}

Simulation::~Simulation() {
	
	delete pt;
	delete glhandler;
	delete vtkwriter;
	delete vtk_pos;

}

// advance simulation to next frame
// should only be called through function pointer
void Simulation::next_frame() {

	update_particles();
	write_pos();

}

// output particle positions to .vtk file
void Simulation::write_pos() {

#ifdef USE_VTK
	// copy positions to host side
	gpu_check(cudaMemcpy(vtk_pos, pt->d_pos, sizeof(float4) * config.n, cudaMemcpyDeviceToHost));
	vtkwriter->write_pos(vtk_pos, config.n);
#endif

}

// advance simulation 
void Simulation::update_particles() {
	
	int grid_size = (config.n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	update_kernel<<<grid_size, BLOCK_SIZE>>>(pt->d_pos, pt->d_vel, pt->d_acc, config.n);
	gpu_check(cudaPeekAtLastError());

	// wait for kernels to finish
	gpu_check(cudaDeviceSynchronize());

}

// calculates accleration vector from forces
__device__
float4 calc_acc(const float4 pos_i, const float4 pos_j, float4 acc) {

	float3 r = {
		pos_j.x - pos_i.x,
		pos_j.y - pos_i.y,
		pos_j.z - pos_i.z
	};
	
	// using softening parameter
	float dist = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
	float inv_distcb = 1.0f / sqrtf(dist * dist * dist);

	acc.x += r.x * inv_distcb;
	acc.y += r.y * inv_distcb;
	acc.z += r.z * inv_distcb;

	return acc;

}

// particle positions per thread tile in 
extern __shared__ float4 s_pos[BLOCK_SIZE];

// calculates updated acceleration vector
__device__
float4 update_acc(const float4* d_pos, const int n) {

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
		for (int j = 0; j < blockDim.x; j++) {
			acc = calc_acc(pos, s_pos[j], acc);
		}
		__syncthreads();

		tile++;

	}

	return acc;

}

// updates particle properties
__global__
void update_kernel(float4* d_pos, float4* d_vel, float4* d_acc, const int n) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	float4 pos = d_pos[id];
	float4 vel = d_vel[id];
	float4 acc = d_acc[id];

	// using kick-drift-kick leapfrog integrator
	float t_2 = TIME_STEP / 2;

	// velocity at mid-point of time step
	float4 vel_mid = make_float4(
		vel.x + acc.x * t_2,
		vel.y + acc.y * t_2,
		vel.z + acc.z * t_2
	);

	// update position
	d_pos[id] = make_float4(
		pos.x + vel_mid.x * TIME_STEP,
		pos.y + vel_mid.y * TIME_STEP,
		pos.z + vel_mid.z * TIME_STEP
	);

	// calculate new accleration
	acc = update_acc(d_pos, n);

	// update velocity
	d_vel[id] = make_float4(
		vel_mid.x + acc.x * t_2,
		vel_mid.y + acc.y * t_2,
		vel_mid.z + acc.z * t_2
	);

	// update accleration
	d_acc[id] = acc;

}