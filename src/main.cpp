//
//	Entry point for program
//

#include <chrono>
#include <iostream>
#include <iomanip>
#include "debug.h"
#include "config.h"
#include "simulation.h"
#include "vtkwriter.h"
#include "glhandler.h"

// temp. control for VTK_Writer
// #define USE_VTK

int main() {

	print_dev_prop();

	// load configuration
	Config config;
	config.load();

#ifdef USE_VTK
	using Clock = std::chrono::high_resolution_clock;
	using time_ms = std::chrono::milliseconds;

	// calculates elapsed time in seconds
	auto get_elapsed = [](auto start, auto end) {
		time_ms dur = std::chrono::duration_cast<time_ms>(end - start);
		long long int ms = dur.count();

		return static_cast<float>(ms) / 1000;
	};

	// simulation start time
	auto start_time = Clock::now();
#endif

	std::cout << "Simulation started" << std::endl;

	Simulation sim(config);
	GLHandler glhandler(&sim);

	// start simulation through graphics handler
	glhandler.loop();

#ifdef USE_VTK
	// simulation end time
	auto end_time = Clock::now();
	
	auto elapsed_time = get_elapsed(start_time, end_time);
	std::cout << std::setprecision(4) << "Simulation ended ["
		<< config.frames << " frames in " << elapsed_time
		<< "; FPS = " << (config.frames / elapsed_time) << "]"
		<< std::endl;
#endif

	getchar();
	return 0;

}