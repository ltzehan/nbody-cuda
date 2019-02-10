//
//	Handles OpenGL inter-op and display
//

#include "glhandler.h"
#include "debug.h"
#include "simulation.h"

// tentative definitions
#define GRPH_WIN_W 640
#define GRPH_WIN_H 480

// argument is a function pointer to advance simulation to the next frame
GLHandler::GLHandler(Simulation* sim) {

	// initialize GLFW
	if (!glfwInit()) {
		// failed
		glfw_error(-1, "Failed to initialize GLFW");

	}
	else {

		// set error callback function
		glfwSetErrorCallback(glfw_error);

		window = glfwCreateWindow(GRPH_WIN_W, GRPH_WIN_H, "nbody", NULL, NULL);
		if (!window) {
			// failed
			glfw_error(-1, "Failed to create window");
			glfwTerminate();

		}

		glfwMakeContextCurrent(window);

		// intialize GLEW
		GLenum err = glewInit();
		if (err != GLEW_OK) {
			// failed 
			fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));

		}

		glfwHideWindow(window);

		// set key action callback function
		glfwSetKeyCallback(window, keys_callback);

	}

}

GLHandler::~GLHandler() {

	glfwTerminate();

}

// display loop
void GLHandler::loop() {

	// make window visible
	glfwShowWindow(window);
	glfwFocusWindow(window);

	while (!glfwWindowShouldClose(window)) {

		glfwPollEvents();
		glfwSwapBuffers(window);

	}

	glfwDestroyWindow(window);

}

// callback function for keys
void GLHandler::keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		// this is bad and should be fixed somewhere else
		glfwSetWindowShouldClose(window, true);
	}

}