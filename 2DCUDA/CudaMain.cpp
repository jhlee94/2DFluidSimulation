#pragma once
// OpenGL 
#include <GL/glew.h>
// Includes
#include <string.h>
#include <iostream>

// SFML/GUI includes
#include <SFML\Graphics.hpp>
#include <SFML\OpenGL.hpp>

#include <SFGUI\SFGUI.hpp>
#include <SFGUI/Widgets.hpp>

#include "Fluid_Kernels.h"
#include "config.h"

#define CLAMP(v, a, b) (a + (v - a) / (b - a))
#define index(i,j) ((i) + (DIM) *(j))

// Global Variable Init
//sources
float *sd, *su, *sv;

//SFML variables
sf::Clock fps_clock, sim_clock;
float current_time, previous_time, frame_count = 0.f, fps = 0.f;
sf::Font* main_font;
int mouseX0 = -10, mouseY0 = -10;

// Cuda Kernels
extern "C" void initCUDA(int size);
extern "C" void freeCUDA();
extern "C" void step(int size, float dt, float viscosity, float diffusion, float kappa, float sigma, int iteration, float* sd,
	float s_v_i,
	float s_v_j,
	float s_d_i,
	float s_d_j,
	float s_d_val,
	float s_u_val,
	float s_v_val);
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
	sf::RenderWindow app_window(sf::VideoMode(WIDTH, HEIGHT), "2D Fluid Simulator GPU", sf::Style::Default, settings);
	app_window.setVerticalSyncEnabled(true);

	main_font = new sf::Font;
	main_font->loadFromFile("../Resources/arial.ttf");

	sfg::SFGUI sfgui;
	app_window.setActive();

	// Initialise CUDA requirements
	int size = DS;

	std::cout << "Max Grid Size: " << size << std::endl;
	std::cout << "Max Tile Size: " << TILE_SIZE_X << std::endl;

	// Initialise Sources
	sd = new float[size];

	for (int i = 0; i < size; i++){
		sd[i] = 0.f;
	}


	float s_v_i, s_v_j, s_d_i, s_d_j, s_d_val, s_u_val, s_v_val;

	// Init Fluid variables on Device side
	initCUDA(size);

	// Init GLEW functions
	glewInit();

	// GUI
	auto viscosity_scale = sfg::Scale::Create(0.f, 0.01f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto diffusion_scale = sfg::Scale::Create(0.f, 0.005f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
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

	viscosity_scale->SetValue(0.f);
	diffusion_scale->SetValue(0.f);
	solver_scale->SetValue(10.f);
	dt_scale->SetValue(0.01f);

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
		s_d_val = 0.f;
		s_u_val = 0.f;
		s_v_val = 0.f;
		auto delta = sim_clock.restart().asSeconds();
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
				desktop.HandleEvent(event);
				if (event.type == sf::Event::MouseMoved)
				{

					int mouseX = event.mouseMove.x;
					int mouseY = event.mouseMove.y;
					if ((mouseX >= 0 && mouseX < WIDTH) && (mouseY >= 0 && mouseY < HEIGHT)){
						s_v_i = (mouseX / static_cast<float>(WIDTH)) * DIM + 1;
						s_v_j = (mouseY / static_cast<float>(HEIGHT)) * DIM + 1;
						float dirX = (mouseX - mouseX0) * 100;
						float dirY = (mouseY - mouseY0) * 100;
						s_u_val = dirX;
						s_v_val = dirY;

						mouseX0 = mouseX;
						mouseY0 = mouseY;
					}
				}
			}
		}

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
			s_d_i = (sf::Mouse::getPosition(app_window).x / static_cast<float>(WIDTH)) * DIM + 1;
			s_d_j = (sf::Mouse::getPosition(app_window).y / static_cast<float>(HEIGHT)) * DIM + 1;
			s_d_val = 50.f;
		}

		step(DIM, 
			//delta,
			dt_scale->GetValue(), 
			viscosity_scale->GetValue(), 
			diffusion_scale->GetValue(), 
			0.3f, 
			0.0f, 
			solver_scale->GetValue(), 
			sd,
			s_v_i,
			s_v_j,
			s_d_i,
			s_d_j,
			s_d_val,
			s_u_val,
			s_v_val);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (int i = 1; i < DIM-1; i++) {
			for (int j = 1; j < DIM-1; j++) {
				int cell_idx = index(i,j);

				float density = sd[cell_idx];
				float color;
				if (density > 0)
				{
					//color = std::fmod(density, 100.f) / 100.f;
					glPushMatrix();
					glTranslatef(i*TILE_SIZE_X, j*TILE_SIZE_Y, 0);
					glBegin(GL_QUADS);
					if (j < DIM - 1)
						applyColor(sd[index(i, j + 1)],
						su[index(i, j + 1)],
						sv[index(i, j + 1)]);
					else
						applyColor(density, su[cell_idx], sv[cell_idx]);
					glVertex2f(0.f, TILE_SIZE_Y);

					applyColor(density, su[cell_idx], sv[cell_idx]);
					glVertex2f(0.f, 0.f);

					if (i < DIM - 1)
						applyColor(sd[index(i + 1, j)],
						su[index(i + 1, j)],
						sv[index(i + 1, j)]);
					else
						applyColor(density, su[cell_idx], sv[cell_idx]);
					glVertex2f(TILE_SIZE_X, 0.f);

					if (i < DIM - 1 && j < DIM - 1)
						applyColor(sd[index(i + 1, j + 1)],
						su[index(i + 1, j + 1)],
						sv[index(i + 1, j + 1)]);
					else
						applyColor(density, su[cell_idx], sv[cell_idx]);
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
	}

	freeCUDA();
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