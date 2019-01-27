#pragma once

#include <inc_filesystem.h>

// parameters are examples for development
// expect more variables controlling the physics to be introduced over time
// frame variable is likely to be removed when OpenGL interop is integrated
struct Config {

	// number of bodies
	int n;
	// number of frames
	int frames;

	bool load();
	void print();

	static void create(fs::path p);

};