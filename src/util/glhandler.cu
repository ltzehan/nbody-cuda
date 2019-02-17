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
GLHandler::GLHandler(Simulation* sim) : sim(sim) {

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
		glfwHideWindow(window);

		// set key action callback function
		glfwSetKeyCallback(window, keys_callback);

		// intialize GLEW
		GLenum err = glewInit();
		if (err != GLEW_OK) {
			// failed 
			fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));

		}

		// set background to black
		glClearColor(0.0, 0.0, 0.0, 1.0);

		// enable depth testing
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		// using whole window for rendering
		glViewport(0, 0, GRPH_WIN_W, GRPH_WIN_H);

		// set up perspective projection
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(45.0, 1.0, 0.1, 100);	// might need tweaking

		// map particles to render space
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();


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

	// compute and redraw
	while (!glfwWindowShouldClose(window)) {

		// advance simulation
		sim->next_frame();

		// draw particles

		// clear buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glfwPollEvents();
		glfwSwapBuffers(window);

	}

	glfwDestroyWindow(window);

}

// callback function for keys
void GLHandler::keys_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}

}

// callback function for resizing
void GLHandler::resize_callback(GLFWwindow* window, int width, int height) {

	glfwSetWindowSize(window, GRPH_WIN_W, GRPH_WIN_H);

}