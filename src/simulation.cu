//
//	Manages control flow of the simulation
//	Will also manage OpenGL visualization when implemented
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simulation.h"
#include "vtkwriter.h"

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
		
		update_particles();

		vtkw.write_pos(pt->d_pos, config.n);

	}

}

// advance simulation 
void Simulation::update_particles() {
	
}
