//
//	Handles OpenGL inter-op and display
//

#include "glhandler.h"
#include "debug.h"

// tentative definitions
#define GRPH_WIN_W 640
#define GRPH_WIN_H 480

GLHandler::GLHandler() {

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

		}

		glfwMakeContextCurrent(window);

		// set key action callback function
		glfwSetKeyCallback(window, keys_callback);

		// intialize GLEW
		GLenum err = glewInit();
		if (err != GLEW_OK) {
			// failed 
			fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));

		}

	}

}

GLHandler::~GLHandler() {

	glfwDestroyWindow(window);
	glfwTerminate();

}

// callback function for keys
void GLHandler::keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		// this is bad and should be fixed somewhere else
		glfwDestroyWindow(window);
	}

}