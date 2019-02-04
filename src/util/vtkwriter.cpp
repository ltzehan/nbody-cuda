//
//	Outputs particle properties in the VTK file format
//	To be used with ParaView for development and testing purposes
//

#include <string>
#include <fstream>
#include <iostream>
#include "vtkwriter.h"
#include "debug.h"

const std::string OUT_DIR = "nbody-out";

VTKWriter::VTKWriter(bool del_old_dirs) {

	// initialize frame counter
	frame = 0;

	// try default output directory
	fs::path p = fs::current_path();
	p /= OUT_DIR;

	try {

		if (fs::exists(p)) {

			if (del_old_dirs) {
				// overwrite previous output directory
				
				for (const auto& f : fs::directory_iterator(p)) {
					fs::remove(f);
				}

			}
			else {
				// create a unique output directory
				int uniq = 1;
				while (fs::exists(p)) {

					p.replace_filename(OUT_DIR + "-" + std::to_string(uniq));
					uniq++;

				}

			}

		}

		fs::create_directory(p);
		outdir = p;

	}
	catch (const fs::filesystem_error& e) {
		std::cout << e.what() << "\n" << "path: " << p << "\n";
	}

}

// write particle positions to file
void VTKWriter::write_pos(const float4* h_pos, const int n) {

	fs::path file = outdir / ("frame-" + std::to_string(frame) + ".vtk");

	std::ofstream ofs(file);
	// VTK file header
	ofs << "# vtk DataFile Version 3.0\n";
	ofs << "Particle Data (Frame " << frame << ")\n";
	ofs << "ASCII\n";
	ofs << "DATASET POLYDATA\n";
	ofs << "POINTS " << n << " float\n";

	// write particle positions
	for (int i = 0; i < n; i++) {
		
		float4 pos = h_pos[i];

		ofs << pos.x << " " << pos.y << " " << pos.z << "\n";

	}

	frame++;

}
