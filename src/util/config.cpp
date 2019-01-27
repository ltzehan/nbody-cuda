//
//	Handles configuration file creation and parsing
//

#include <fstream>
#include <iostream>
#include <iomanip>
#include "config.h"

// default parameters
const int DEFAULT_N = 2048;
const int DEFAULT_FRAMES = 50;

// configuration file tokens
const char CONFIG_COMMENT = '#';
const char CONFIG_DELIMITER = '=';

// configuration file keys
const std::string CONFIG_KEY_N = "N";
const std::string CONFIG_KEY_FRAMES = "FRAMES";

// file name
const std::string CONFIG_FILE = "nbody.cfg";

// methods for input validation
namespace {

	// removes whitespace from string
	void remove_whitespace(std::string& str) {
		str.erase(
			std::remove(str.begin(), str.end(), ' '),
			str.end()
		);
	}

	// called when parsing encounters error
	void print_parse_err(std::string err, std::string key = "") { 
		std::cout << "Error when parsing configuration file:\n" 
			<< err << " " << key << std::endl; 
	}

	// assigns value to reference passed if the given string represents
	// a positive non-zero value
	bool get_posnz_int(std::string str, int& ref) {

		try {
			size_t c;
			int x = std::stoi(str, &c);
			if (x > 0 && c == str.size()) {
				ref = x;
				return true;
			}
			else {
				// negative or float
				return false;
			}
		}
		catch (const std::invalid_argument&) {
			return false;
		}

	}
	// associated error
	void print_not_posnz_int(std::string key) {
		print_parse_err("Expected positive non-zero integer for key", key); 
	}

	void print_unrecog_key(std::string key) {
		print_parse_err("Unrecognized key", key);
	}

}

// tries to load simulation parameters from configuration file
// returns true if loading was successful
bool Config::load() {

	// initialize with default parameters
	this->n = DEFAULT_N;
	this->frames = DEFAULT_FRAMES;

	// try finding configuration file from current directory
	fs::path p = fs::current_path();
	p /= CONFIG_FILE;

	if (!fs::exists(p)) {
		// create new file
		std::cout << "Creating new configuration file...\n";
		create(p);

		std::cout << "Using default parameters:\n";
		this->print();
	}
	else {
		// read from file
		std::ifstream ifs(p);

		std::string line;
		while (std::getline(ifs, line)) {

			// ignore comments and blank lines
			if (line.size() > 0 && line.at(0) != CONFIG_COMMENT) {
				
				remove_whitespace(line);

				// position of delimiter
				int pos = line.find(CONFIG_DELIMITER);

				std::string	key = line.substr(0, pos);
				std::string val = line.substr(pos + 1);

				// validate values according to key
				if (key == CONFIG_KEY_N) {

					if (!get_posnz_int(val, this->n)) {
						print_not_posnz_int(key);
						return false;
					}
				}
				else if (key == CONFIG_KEY_FRAMES) {

					if (!get_posnz_int(val, this->frames)) {
						print_not_posnz_int(key);
						return false;
					}
				}
				else {
					// unrecognized key
					print_unrecog_key(key);
					return false;
				}
			}

		}

		std::cout << "Parsed parameters from configuration file:\n";
		this->print();
	}

	return true;
}

// prints out current simulation parameters
void Config::print() {

	std::cout << std::left << std::setfill(' ')
		<< std::setw(8) << "N" << " = " << this->n << "\n"
		<< std::setw(8) << "FRAMES" << " = " << this->frames << "\n"
		<< std::endl;
	 
}

// create new configuration file and populate with default parameters
void Config::create(fs::path p) {

	std::ofstream ofs(p);
	ofs << "# Configuration file for nbody\n"
		<< "# Parameters are set using key-value pairs\n"
		<< "# Default values will be used if not explicitly specified"
		<< "\n"
		<< "# Number of bodies\n"
		<< "N=" << DEFAULT_N << "\n"
		<< "\n"
		<< "# Number of frames\n"
		<< "FRAMES=" << DEFAULT_FRAMES << "\n";

}