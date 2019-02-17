#pragma once

#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct Simulation;

struct GLHandler {

	GLHandler(Simulation* sim);
	~GLHandler();

	void loop();

private:

	Simulation* sim;

	GLFWwindow* window;

	static void keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void resize_callback(GLFWwindow* window, int width, int height);

};