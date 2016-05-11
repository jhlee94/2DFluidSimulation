#pragma once
// OpenGL 
#include <GL/glew.h>
// Includes
#include <string.h>
#include <iostream>
// SFML/GUI includes
#include <SFML\Graphics.hpp>
#include <SFML\OpenGL.hpp>

#include "Fluid_Kernels.h"
#include "FluidPanel.h"
#include "config.h"

// Global Variable Init
//sources
float *sd;
uchar4 *h_textureBufferData;
uchar4 *d_textureBufferData;

//SFML variables
sf::Clock fps_clock, sim_clock;
float current_time, previous_time = 0.f, frame_count = 0.f, fps = 0.f;
sf::Font* main_font;
sf::Text fps_text;
int mouseX0 = -10, mouseY0 = -10;
float s_v_i, s_v_j, s_d_i, s_d_j, s_d_val, s_u_val, s_v_val;
sf::RenderWindow *app_window;

//GUI
FluidPanel* panel;
bool gui = false;

// vbo variables
GLuint vbo = 0;
GLuint indices_va = 0;
GLuint pbo;
struct cudaGraphicsResource *cudaPBO;
GLuint tex;

// Cuda Kernels
extern "C" void initCUDA(int size);
extern "C" void freeCUDA();
extern "C" void step(int size,
	Parameters &parameters,
	float *sd,
	float s_v_i,
	float s_v_j,
	float s_d_i,
	float s_d_j,
	float s_d_val,
	float s_u_val,
	float s_v_val);
extern "C" void createTexture(int size, uchar4* d_texture);


// Graphics Functions
void InitSFML();
bool InitGL(int width, int height);
void CreateFrame();
void CalculateFPS(void);

void Display();
void HandleInput();
void Clean();

void DrawGrid(bool);
void PrintString(float x, float y, sf::Text& text, const char* string, ...);
void ApplyColour(float x, float, float);

int main(void)
{
	InitSFML();
	// Initialise CUDA requirements
	int size = DS;
	std::cout << "Max Grid Size: " << size << std::endl;
	std::cout << "Max Tile Size: " << TILE_SIZE_X << std::endl;
	initCUDA(size);
	// Init GLEW functions
	glewInit();
	InitGL(DIM, DIM);

	//// Initialise Sources
	//sd = new float[size];

	//for (int i = 0; i < size; i++){
	//	sd[i] = 0.f;
	//}

	// Init Fluid variables on Device side
	//SFML mainloop
	while (app_window->isOpen()) {
		CalculateFPS();
		s_d_val = 0.f;
		s_u_val = 0.f;
		s_v_val = 0.f;
		HandleInput();
		step(DIM,
			//delta,
			panel->m_parameters,
			sd,
			s_v_i,
			s_v_j,
			s_d_i,
			s_d_j,
			s_d_val,
			s_u_val,
			s_v_val);
		Display();
		// Finally, Display all
		app_window->display();
		//glFlush();
	}
	Clean();
	return 0;
}

void DrawGrid(bool x)
{
	if (x)
	{
		glColor4f(0.f, 1.f, 0.f, 1.f);
		for (float x = (static_cast<float>(WIDTH) / DIM) / static_cast<float>(WIDTH); x < 1; x += (static_cast<float>(WIDTH) / DIM) / static_cast<float>(WIDTH)){
			glBegin(GL_LINES);
			glVertex2f(0, x);
			glVertex2f(1, x);
			glEnd();
		};
		for (float y = (static_cast<float>(HEIGHT) / DIM) / static_cast<float>(HEIGHT); y < 1; y += (static_cast<float>(HEIGHT) / DIM) / static_cast<float>(HEIGHT)){
			glBegin(GL_LINES);
			glVertex2f(y, 0);
			glVertex2f(y, 1);
			glEnd();
		};
	}
}

void Display() {
	auto delta = sim_clock.restart().asSeconds();
	CreateFrame();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	glColor3f(1.0f, 1.0f, 1.0f);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glBegin(GL_QUADS);
	glTexCoord2f(0.f, 0.f);	glVertex2f(0.f, 0.f);
	glTexCoord2f(0.f, 1.f);	glVertex2f(0.f, 1.f);
	glTexCoord2f(1.f, 1.f);	glVertex2f(1.f, 1.f);
	glTexCoord2f(1.f, 0.f);	glVertex2f(1.f, 0.f);
	glEnd();

	// Release
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glPopMatrix();

	glPushMatrix();
	glColor3f(1.0f, 1.0f, 1.0f);

	// GL Immediate Drawing
	// Need cudaMemcpy of Density field from the Device to Host
	//for (int i = 1; i < DIM-1; i++) {
	//	for (int j = 1; j < DIM-1; j++) {
	//		int cell_idx = index(i,j);

	//		float density = sd[cell_idx];
	//		float color;
	//		if (density > 0)
	//		{
	//			//color = std::fmod(density, 100.f) / 100.f;
	//			glPushMatrix();
	//			glTranslatef(i*TILE_SIZE_X, j*TILE_SIZE_Y, 0);
	//			glBegin(GL_QUADS);
	//			ApplyColour(density, su[cell_idx], sv[cell_idx]);
	//			glVertex2f(0.f, TILE_SIZE_Y);
	//			glVertex2f(0.f, 0.f);
	//			glVertex2f(TILE_SIZE_X, 0.f);
	//			glVertex2f(TILE_SIZE_X, TILE_SIZE_Y);
	//			glEnd();
	//			glPopMatrix();
	//		}
	//	}
	//}

	// Grid Lines 
	glPushMatrix();
	DrawGrid(panel->m_parameters.grid);
	glPopMatrix();

	// SFML rendering.
	// Draw FPS Text
	app_window->pushGLStates();
	PrintString(5, 16, fps_text, "FPS: %5.2f", fps);
	app_window->draw(fps_text);
	// SFGUI Update
	panel->Update(delta);
	panel->Display(*app_window);
	app_window->popGLStates();

}

void HandleInput()
{
	sf::Event event;
	while (app_window->pollEvent(event)) {
		if (event.type == sf::Event::Closed) {
			app_window->close();
			break;
		}
		else if (event.type == sf::Event::Resized) {
			glViewport(0, 0, event.size.width, event.size.height);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, 1, 1, 0, 0, 1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
		}
		else if (event.type == sf::Event::LostFocus)
		{
			// Pause the system
		}
		else {
			panel->HandleEvent(event);
			if (event.type == sf::Event::MouseMoved)
			{

				int mouseX = event.mouseMove.x;
				int mouseY = event.mouseMove.y;
				if ((mouseX >= 0 && mouseX < WIDTH) && (mouseY >= 0 && mouseY < HEIGHT)){
					s_v_i = (mouseX / static_cast<float>(WIDTH)) * DIM;
					s_v_j = (mouseY / static_cast<float>(HEIGHT)) * DIM;
					float dirX = (mouseX - mouseX0) * 300;
					float dirY = (mouseY - mouseY0) * 300;
					s_u_val = dirX;
					s_v_val = dirY;

					mouseX0 = mouseX;
					mouseY0 = mouseY;
				}
			}
		}
	}

	if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
		s_d_i = (sf::Mouse::getPosition(*app_window).x / static_cast<float>(WIDTH)) * DIM;
		s_d_j = (sf::Mouse::getPosition(*app_window).y / static_cast<float>(HEIGHT)) * DIM;
		s_d_val = 500.f;
	}
}

void PrintString(float x, float y, sf::Text& text, const char* string, ...)
{
	char buffer[128];
	va_list arg;
	_crt_va_start(arg, string);
	vsprintf(buffer, string, arg);
	_crt_va_end(arg);

	if (!text.getFont())
		text.setFont(*main_font);
	text.setCharacterSize(15);
	text.setPosition(x, y);
	text.setString(buffer);
	text.setColor(sf::Color::Green);
}

void CalculateFPS()
{
	//  Increase frame count
	frame_count++;

	//  Get the number of milliseconds since glutInit called
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	current_time = fps_clock.getElapsedTime().asMilliseconds();
	//  Calculate time passed
	int timeInterval = current_time - previous_time;

	if (timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frame_count / (timeInterval / 1000.0f);

		//  Set time
		previous_time = current_time;
		//  Reset frame count
		frame_count = 0;

	}
}

void ApplyColour(float x, float, float){
	const float treshold1 = 1.;
	const float treshold2 = 4.;
	const float treshold3 = 10.;

	/* red */
	if (x < treshold1) {
		glColor4f(CLAMP(x, 0., treshold1), 0., 0., 0.8);
	}

	/* yellow */
	else if (x < treshold2) {
		glColor4f(1., CLAMP(x, treshold1, treshold2) - treshold1, 0., 0.8);
	}

	/* white */
	else if (x < treshold3){
		glColor4f(1., 1., CLAMP(x, treshold2, treshold3) - treshold2, 0.8);
	}

	else{
		glColor4f(1., 1., 1., 1.);
	}

}

bool InitGL(int width, int height) {

	// Allocate new buffers.
	h_textureBufferData = new uchar4[width * height];

	glEnable(GL_TEXTURE_2D);

	// Unbind any textures from previous.
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);/*
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);*/
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, h_textureBufferData);

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4),
		h_textureBufferData, GL_STREAM_COPY);

	cudaError result = cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo,
		cudaGraphicsMapFlagsWriteDiscard);

	glUniform1i(tex, 0);
	// unbind
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// GL_Display Init
	glViewport(0, 0, static_cast<int>(app_window->getSize().x), static_cast<int>(app_window->getSize().y));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0, 1, 1, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	return result == cudaSuccess;
}

void CreateFrame() {
	cudaGraphicsMapResources(1, &cudaPBO, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData,
		&num_bytes, cudaPBO);

	createTexture(DIM, d_textureBufferData);

	cudaGraphicsUnmapResources(1, &cudaPBO, 0);
}

void InitSFML() {
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	app_window = new sf::RenderWindow(sf::VideoMode(WIDTH, HEIGHT), "2D Fluid Simulator GPU", sf::Style::Default, settings);
	//app_window->setVerticalSyncEnabled(true);

	main_font = new sf::Font;
	main_font->loadFromFile("../Resources/arial.ttf");

	panel = new FluidPanel(&gui);
	panel->Initialise();
	app_window->setActive();
}

void Clean()
{
	freeCUDA();
	delete main_font;
	delete[] sd;
	delete h_textureBufferData;
	delete panel;
	h_textureBufferData = nullptr;
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &pbo);
	glDeleteBuffers(1, &indices_va);
	glDeleteTextures(1, &tex);
	delete app_window;
}