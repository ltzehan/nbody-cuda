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
