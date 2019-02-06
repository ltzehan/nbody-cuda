#pragma once

#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>

struct GLHandler {

	GLHandler();
	~GLHandler();

private:

	GLFWwindow* window;

	static void keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

};