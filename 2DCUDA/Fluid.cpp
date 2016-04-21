#pragma once
// OpenGL 
#include <GL/glew.h>
// Includes
#include <string.h>
#include <iostream>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// SFML/GUI includes
#include <SFML\Graphics.hpp>
#include <SFML\OpenGL.hpp>

#include <SFGUI\SFGUI.hpp>
#include <SFGUI/Widgets.hpp>

#include "config.h"
#include "Fluid_Kernels.h"
#define CLAMP(v, a, b) (a + (v - a) / (b - a))

//sources
float *sd, *su, *sv;

//SFML variables
sf::Clock fps_clock, sim_clock;
float current_time, previous_time, frame_count = 0.f, fps = 0.f;
sf::Font* main_font;
int mouseX0 = -10, mouseY0 = -10;

// Cuda Kernels

// Graphics Functions
void DrawGrid(bool);
void PrintString(float x, float y, sf::Text& text, const char* string, ...);
void CalculateFPS(void);
void applyColor(float x, float, float);

int main(void)
{
	// An sf::Window for raw OpenGL rendering.
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::RenderWindow app_window(sf::VideoMode(WIDTH, HEIGHT), "2D Fluid Simulator", sf::Style::Default, settings);
	//app_window.setVerticalSyncEnabled(true);

	main_font = new sf::Font;
	main_font->loadFromFile("../Resources/arial.ttf");

	sfg::SFGUI sfgui;
	app_window.setActive();

	// Initialise CUDA requirements

	std::cout << DS << std::endl;
	std::cout << TILE_SIZE_X << std::endl;

	// Initialise Sources
	sd = new float[DS];
	su = new float[DS];
	sv = new float[DS];

	for (int i = 0; i < DS; i++){
		sd[i] = 0.f;
		su[i] = 0.f;
		sv[i] = 0.f;
	}

	initCUDA(DS);
	// Init GLEW functions
	glewInit();

	// GUI
	auto viscosity_scale = sfg::Scale::Create(0.f, 0.001f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto diffusion_scale = sfg::Scale::Create(0.f, 0.001f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto solver_scale = sfg::Scale::Create(0.f, 40.f, 1.0f, sfg::Scale::Orientation::HORIZONTAL);
	auto dt_scale = sfg::Scale::Create(0.f, 0.5f, .01f, sfg::Scale::Orientation::HORIZONTAL);
	auto grid_check = sfg::CheckButton::Create("Show Grid");

	auto table = sfg::Table::Create();
	table->SetRowSpacings(5.f);
	table->SetColumnSpacings(5.f);

	table->Attach(sfg::Label::Create("Change the color of the rect using the scales below."), sf::Rect<sf::Uint32>(0, 0, 3, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(sfg::Label::Create("Viscosity:"), sf::Rect<sf::Uint32>(0, 1, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(viscosity_scale, sf::Rect<sf::Uint32>(1, 1, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);
	table->Attach(sfg::Label::Create("Diffusion:"), sf::Rect<sf::Uint32>(0, 2, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(diffusion_scale, sf::Rect<sf::Uint32>(1, 2, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);
	table->Attach(sfg::Label::Create("Solver Iteration:"), sf::Rect<sf::Uint32>(0, 3, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(solver_scale, sf::Rect<sf::Uint32>(1, 3, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);
	table->Attach(sfg::Label::Create("Time Step:"), sf::Rect<sf::Uint32>(0, 4, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(dt_scale, sf::Rect<sf::Uint32>(1, 4, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);
	table->Attach(grid_check, sf::Rect<sf::Uint32>(1, 5, 1, 1), sfg::Table::FILL, sfg::Table::FILL);

	auto window = sfg::Window::Create();
	window->SetTitle("Fluid Panel");
	window->SetPosition(sf::Vector2f(WIDTH - 450.f, 100.f));
	window->Add(table);

	sfg::Desktop desktop;
	desktop.Add(window);

	viscosity_scale->SetValue(.0001f);
	diffusion_scale->SetValue(0.0002f);
	solver_scale->SetValue(20.f);
	dt_scale->SetValue(0.1f);

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
		auto delta = sim_clock.restart().asSeconds();
		sf::Event event;
		while (app_window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				app_window.close();
				freeCUDA();
				break;
			}
			else if (event.type == sf::Event::Resized) {
				glViewport(0, 0, event.size.width, event.size.height);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glOrtho(0, 1, 0, 1, 0, 1);
				glMatrixMode(GL_MODELVIEW);
				glLoadIdentity();
			}
			else if (event.type == sf::Event::LostFocus)
			{
				// Pause the system
			}
			else {
				desktop.HandleEvent(event);
				if (event.type == sf::Event::MouseButtonPressed)
				{
					int i = (event.mouseButton.x / static_cast<float>(WIDTH)) * DIM + 1;
					int j = (event.mouseButton.y / static_cast<float>(HEIGHT)) * DIM + 1;
				}

				if (event.type == sf::Event::MouseMoved)
				{

					int mouseX = event.mouseMove.x;
					int mouseY = event.mouseMove.y;
					if ((mouseX > 0 && mouseX < WIDTH) && (mouseY > 0 && mouseY < HEIGHT)){
						int i = (mouseX / static_cast<float>(WIDTH)) * DIM + 1;
						int j = (mouseY / static_cast<float>(HEIGHT)) * DIM + 1;
						float dirX = (mouseX - mouseX0) * 300;
						float dirY = (mouseY - mouseY0) * 300;

						su[((i + 1) + DIM * j)] = dirX;
						su[(i + DIM * j)] = dirX;
						su[((i - 1) + DIM * j)] = dirX;
						sv[((i + 1) + DIM * j)] = dirY;
						sv[(i + DIM * j)] = dirY;
						sv[((i - 1) + DIM * j)] = dirY;

						mouseX0 = mouseX;
						mouseY0 = mouseY;
					}
				}
			}
		}

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
			int i = (sf::Mouse::getPosition(app_window).x / static_cast<float>(WIDTH)) * DIM + 1;
			int j = (sf::Mouse::getPosition(app_window).y / static_cast<float>(HEIGHT)) * DIM + 1;
		
			sd[((i+1) + DIM * j)] = 100.f;
			sd[(i + DIM * j)] = 100.f;
			sd[((i-1) + DIM * j)] = 100.f;
		}

		step(DIM, 0.01f, 0.f, 0.f, 10, sd, su, sv);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render Density
		for (int i = 1; i < DIM-1; i++) {
			for (int j = 1; j < DIM-1; j++) {
				int cell_idx = i + DIM * j;

				float density = sd[cell_idx];
				float color;
				if (density > 0)
				{
					//color = std::fmod(density, 100.f) / 100.f;
					glPushMatrix();
					glTranslatef(i*TILE_SIZE_X, j*TILE_SIZE_Y, 0);
					glBegin(GL_QUADS);
					applyColor(density, su[cell_idx], sv[cell_idx]);
					glVertex2f(0.f, TILE_SIZE_Y);
					glVertex2f(0.f, 0.f);
					glVertex2f(TILE_SIZE_X, 0.f);
					glVertex2f(TILE_SIZE_X, TILE_SIZE_Y);
					glEnd();
					glPopMatrix();
				}
			}
		}

		// Grid Lines 
		DrawGrid(grid_check->IsActive());

		// SFML rendering.
		// Draw FPS Text
		app_window.pushGLStates();
		PrintString(5, 16, fps_text, "FPS: %5.2f", fps);
		app_window.draw(fps_text);
		// SFGUI Update
		desktop.Update(delta);
		sfgui.Display(app_window);
		app_window.popGLStates();
		// Finally, Display all
		app_window.display();
		//glFlush();
		for (int i = 0; i < DS; i++){
			sd[i] = 0.f;
			su[i] = 0.f;
			sv[i] = 0.f;
		}
	}
	delete main_font;
	delete[] sd, sv, su;
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

void applyColor(float x, float, float){
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