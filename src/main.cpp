//
//	Entry point for program
//

#include <chrono>
#include <iostream>
#include <iomanip>
#include "config.h"
#include "simulation.h"
#include "vtkwriter.h"

#define ENABLE_GPU_CHECK

int main() {

	using Clock = std::chrono::high_resolution_clock;
	using time_ms = std::chrono::milliseconds;

	// load configuration
	Config config;
	config.load();

	// calculates elapsed time in seconds
	auto get_elapsed = [](auto start, auto end) {
		time_ms dur = std::chrono::duration_cast<time_ms>(end - start);
		long long int ms = dur.count();

		return static_cast<float>(ms) / 1000;
	};

	// simulation start time
	auto start_time = Clock::now();
	
	std::cout << "Simulation started" << std::endl;

	Simulation sim(config);
	sim.start();

	// simulation end time
	auto end_time = Clock::now();
	
	auto elapsed_time = get_elapsed(start_time, end_time);
	std::cout << std::setprecision(4) << "Simulation ended ["
		<< config.frames << " frames in " << elapsed_time
		<< "; FPS = " << (config.frames / elapsed_time) << "]"
		<< std::endl;

	getchar();
	return 0;

}