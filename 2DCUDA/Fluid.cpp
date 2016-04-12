// OpenGL 
#include <GL/glew.h>
// Includes
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// SFML/GUI includes
#include <SFML\Graphics.hpp>
#include <SFML\OpenGL.hpp>

#include <SFGUI\SFGUI.hpp>

#include "config.h"
#include "Fluid_Kernels.h"

// Global Variable Init
//velocity and pressure
float *u, *v, *p;
float *uold, *vold, *pold;
float *utemp, *vtemp, *ptemp; // temp pointers, DO NOT INITIALIZE

//divergence of velocity
float *divg;

//density
float *d, *dold, *dtemp;
float *map;

//sources
float *sd, *su, *sv;

//SFML variables
sf::Clock fps_clock, sim_clock;
float current_time, previous_time, frame_count = 0.f, fps = 0.f;
sf::Font* main_font;
int mouseX0 = -10, mouseY0 = -10;

// Cuda Kernels
extern "C" void add_source(float *d, float *s);
extern "C" void advect(float *dold, float *d, float *u, float *v, float md);
extern "C" void divergence(float *u, float *v, float *div);
extern "C" void pressure(float *u, float *v, float *p, float *pold, float *div);
extern "C" void set_bnd(float *u, float *v, float *p);
extern "C" void velocity_bc(float *u, float *v);
extern "C" void pressure_bc(float *p);

// Graphics Functions
void DrawGrid(bool);
void PrintString(float x, float y, sf::Text& text, const char* string, ...);
void CalculateFPS(void);

int main(void)
{
	// An sf::Window for raw OpenGL rendering.
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::RenderWindow app_window(sf::VideoMode(WIDTH, HEIGHT), "2D Fluid Simulator", sf::Style::Default, settings);
	//app_window.setVerticalSyncEnabled(true);

	main_font = new sf::Font;
	main_font->loadFromFile("../Resources/arial.ttf");

	app_window.setActive();

	// Init GLEW functions
	glewInit();
	// GL_Display Init
	glViewport(0, 0, static_cast<int>(app_window.getSize().x), static_cast<int>(app_window.getSize().y));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// FPS init
	sf::Text fps_text;
	previous_time = fps_clock.getElapsedTime().asMilliseconds();

	//SFML mainloop
	while (app_window.isOpen()) {
		CalculateFPS();

		sf::Event event;
		while (app_window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				app_window.close();
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
				if (event.type == sf::Event::MouseButtonPressed)
				{

					int i = (event.mouseButton.x / static_cast<float>(WIDTH)) * DIM + 1;
					int j = (event.mouseButton.y / static_cast<float>(HEIGHT)) * DIM + 1;
				}

				if (event.type == sf::Event::MouseMoved)
				{

					int mouseX = event.mouseMove.x;
					int mouseY = event.mouseMove.y;
					if ((mouseX >= 0 && mouseX < WIDTH) && (mouseY >= 0 && mouseY < HEIGHT)){
						int i = (mouseX / static_cast<float>(WIDTH)) * DIM + 1;
						int j = (mouseY / static_cast<float>(HEIGHT)) * DIM + 1;



						mouseX0 = mouseX;
						mouseY0 = mouseY;
					}
				}
			}
		}

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
			int i = (sf::Mouse::getPosition(app_window).x / static_cast<float>(WIDTH)) * DIM + 1;
			int j = (sf::Mouse::getPosition(app_window).y / static_cast<float>(HEIGHT)) * DIM + 1;
			
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//// Particles
		//glPushMatrix();
		//DrawParticles(red_scale->GetValue(), green_scale->GetValue(), blue_scale->GetValue(), alpha_scale->GetValue());
		//glPopMatrix();

		// Render Density
		

		// Grid Lines 
		DrawGrid(true);

		// SFML rendering.
		// Draw FPS Text
		app_window.pushGLStates();
		PrintString(5, 16, fps_text, "FPS: %5.2f", fps);
		app_window.draw(fps_text);
		//// SFGUI Update
		//desktop.Update(delta);
		//sfgui.Display(app_window);
		app_window.popGLStates();

		// Finally, Display all
		app_window.display();
		//glFlush();
	}

	// cleanup
	delete main_font;
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
	text.setColor(sf::Color::White);
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