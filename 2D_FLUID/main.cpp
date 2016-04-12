#pragma warning (disable:4996)
#include <ctime>
#include <SFGUI\SFGUI.hpp>
#include <SFGUI/Widgets.hpp>

#include <GL\glew.h>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>

#include "Fluid.h"
#define CLAMP(v, a, b) (a + (v - a) / (b - a))

char* ref_file = NULL;

GLuint vbo = 0;
sf::Clock fps_clock, sim_clock;
float current_time, previous_time, frame_count = 0.f, fps = 0.f;
sf::Font* main_font;

int mouseX0 = 0, mouseY0 = 0;

// FluidSolver
Fluid2DCPU* fluid_solver;
Vector2F* particles = NULL;

void DrawGrid(bool);
void CreateGUI(sfg::Desktop& desktop);
void InitParticles(Vector2F *p, int dx, int dy);
void DrawParticles(float r, float g, float b, float a = 1.f);
void PrintString(float x, float y, sf::Text& text, const char* string, ...);
void CalculateFPS(void);
void applyColor(float x, float, float);

float myrand(void);

int main()
{
	// An sf::Window for raw OpenGL rendering.
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::RenderWindow app_window(sf::VideoMode(WIDTH, HEIGHT), "2D Fluid Simulator", sf::Style::Default, settings);
	//app_window.setVerticalSyncEnabled(true);
	main_font = new sf::Font;
	main_font->loadFromFile("../Resources/arial.ttf");
	fluid_solver = new Fluid2DCPU(TILE_DIM);
	// Create an SFGUI. This is required before doing anything with SFGUI.
	sfg::SFGUI sfgui;
	// Set the SFML Window's context back to the active one. SFGUI creates
	// a temporary context on creation that is set active.
	app_window.setActive();

	// Define GUI
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
	window->SetPosition(sf::Vector2f(WIDTH-450.f, 100.f));
	window->Add(table);

	sfg::Desktop desktop;
	desktop.Add(window);

	viscosity_scale->SetValue(.0001f);
	diffusion_scale->SetValue(0.0002f);
	solver_scale->SetValue(20.f);
	dt_scale->SetValue(0.1f);

	// Init GLEW functions
	glewInit();
	// GL_Display Init
	glViewport(0, 0, static_cast<int>(app_window.getSize().x), static_cast<int>(app_window.getSize().y));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	// Init Particles
	particles = new Vector2F[(DIM*DIM)];
	InitParticles(particles, DIM, DIM);

	// FPS init
	sf::Text fps_text;
	previous_time = fps_clock.getElapsedTime().asMilliseconds();

	// SFML mainloop
	sf::Event event;

	while (app_window.isOpen()) {
		CalculateFPS();
		// Simulate Fluid Solver step
		auto delta = sim_clock.restart().asSeconds();
		

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
				//if ()
				desktop.HandleEvent(event);
				if (event.type == sf::Event::MouseButtonPressed)
				{
					
					int i = (event.mouseButton.x / static_cast<float>(WIDTH)) * TILE_DIM + 1;
					int j = (event.mouseButton.y / static_cast<float>(HEIGHT)) * TILE_DIM + 1;
				}

				if (event.type == sf::Event::MouseMoved)
				{
					
					int mouseX = event.mouseMove.x;
					int mouseY = event.mouseMove.y;
					if ((mouseX >= 0 && mouseX < WIDTH) && (mouseY >= 0 && mouseY < HEIGHT)){
						int i = (mouseX / static_cast<float>(WIDTH)) * TILE_DIM + 1;
						int j = (mouseY / static_cast<float>(HEIGHT)) * TILE_DIM + 1;
						float dirX = (mouseX - mouseX0) * 1.5;
						float dirY = (mouseY - mouseY0) * 1.5;
						fluid_solver->u_prev[fluid_solver->index(i, j)] = dirX;
						fluid_solver->v_prev[fluid_solver->index(i, j)] = dirY;


						fluid_solver->u_prev[fluid_solver->index(i + 1, j)] = dirX;
						fluid_solver->v_prev[fluid_solver->index(i + 1, j)] = dirY;
						fluid_solver->u_prev[fluid_solver->index(i - 1, j)] = dirX;
						fluid_solver->v_prev[fluid_solver->index(i - 1, j)] = dirY;
						fluid_solver->u_prev[fluid_solver->index(i, j + 1)] = dirX;
						fluid_solver->v_prev[fluid_solver->index(i, j + 1)] = dirY;
						fluid_solver->u_prev[fluid_solver->index(i, j - 1)] = dirX;
						fluid_solver->v_prev[fluid_solver->index(i, j - 1)] = dirY;



						mouseX0 = mouseX;
						mouseY0 = mouseY;
					}
				}
			}
		}

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)){
			int i = (sf::Mouse::getPosition(app_window).x / static_cast<float>(WIDTH)) * TILE_DIM + 1;
			int j = (sf::Mouse::getPosition(app_window).y / static_cast<float>(HEIGHT)) * TILE_DIM + 1;
			fluid_solver->dens_prev[fluid_solver->index(i, j)] = 100.f;
			//fluid_solver->dens_prev[fluid_solver->index(i - 1, j)] = 50.f;
			//fluid_solver->dens_prev[fluid_solver->index(i + 1, j)] = 50.f;

			fluid_solver->u_prev[fluid_solver->index(i, j)] = 0.f;
			fluid_solver->v_prev[fluid_solver->index(i, j)] = -3.0f;
			fluid_solver->u_prev[fluid_solver->index(i + 1, j)] = 0.f;
			fluid_solver->v_prev[fluid_solver->index(i + 1, j)] = -3.0f;
			fluid_solver->u_prev[fluid_solver->index(i - 1, j)] = 0.f;
			fluid_solver->v_prev[fluid_solver->index(i - 1, j)] = -3.0f;
			fluid_solver->u_prev[fluid_solver->index(i, j + 1)] = 0.f;
			fluid_solver->v_prev[fluid_solver->index(i, j + 1)] = -3.0f;
			fluid_solver->u_prev[fluid_solver->index(i, j - 1)] = 0.f;
			fluid_solver->v_prev[fluid_solver->index(i, j - 1)] = -3.0f;
		}
		
		fluid_solver->m_viscosity = viscosity_scale->GetValue();
		fluid_solver->m_diffusion = diffusion_scale->GetValue();
		fluid_solver->iteration = solver_scale->GetValue();

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

		//// Particles
		//glPushMatrix();
		//DrawParticles(red_scale->GetValue(), green_scale->GetValue(), blue_scale->GetValue(), alpha_scale->GetValue());
		//glPopMatrix();
		
		// Render Density
		fluid_solver->step(dt_scale->GetValue());
		for (int i = 1; i <= TILE_DIM; i++) {
			for (int j = 1; j <= TILE_DIM; j++) {
				int cell_idx = fluid_solver->index(i, j);

				float density = fluid_solver->dens[cell_idx];
				float color;
				if (density > 0)
				{
					//color = std::fmod(density, 100.f) / 100.f;
					glPushMatrix();
					glTranslatef(i*TILE_SIZE_X - TILE_SIZE_X, j*TILE_SIZE_Y - TILE_SIZE_Y, 0);
					glBegin(GL_QUADS);
					
					if (j < TILE_DIM - 1)
						applyColor(fluid_solver->dens[fluid_solver->index(i, j+1)], 
								   fluid_solver->u[fluid_solver->index(i, j+1)], 
								   fluid_solver->v[fluid_solver->index(i, j+1)]);
					else
						applyColor(density, fluid_solver->u[cell_idx], fluid_solver->v[cell_idx]);
					glVertex2f(0.f, TILE_SIZE_Y);
					
					applyColor(density, fluid_solver->u[cell_idx], fluid_solver->v[cell_idx]);
					glVertex2f(0.f, 0.f);

					if (i < TILE_DIM - 1)
						applyColor(fluid_solver->dens[fluid_solver->index(i + 1, j)], 
								   fluid_solver->u[fluid_solver->index(i + 1, j)], 
								   fluid_solver->v[fluid_solver->index(i + 1, j)]);
					else
						applyColor(density, fluid_solver->u[cell_idx], fluid_solver->v[cell_idx]);
					glVertex2f(TILE_SIZE_X, 0.f);
					if (i < TILE_DIM - 1 && j < TILE_DIM - 1)
						applyColor(fluid_solver->dens[fluid_solver->index(i + 1, j + 1)],
						fluid_solver->u[fluid_solver->index(i + 1, j + 1)],
						fluid_solver->v[fluid_solver->index(i + 1, j + 1)]);
					else
						applyColor(density, fluid_solver->u[cell_idx], fluid_solver->v[cell_idx]);
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

	// cleanup
	delete[] particles;
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDeleteBuffersARB(1, &vbo);
	delete main_font;
	delete fluid_solver;
	return 0;
}

void Display()
{
	
}

void CreateGUI()
{
}

void DrawGrid(bool x)
{
	if (x)
	{
		glColor4f(0.f, 1.f, 0.f, 1.f);
		for (float x = (static_cast<float>(WIDTH) / TILE_DIM) / static_cast<float>(WIDTH); x < 1; x += (static_cast<float>(WIDTH) / TILE_DIM) / static_cast<float>(WIDTH)){
			glBegin(GL_LINES);
			glVertex2f(0, x);
			glVertex2f(1, x);
			glEnd();
		};
		for (float y = (static_cast<float>(HEIGHT) / TILE_DIM) / static_cast<float>(HEIGHT); y < 1; y += (static_cast<float>(HEIGHT) / TILE_DIM) / static_cast<float>(HEIGHT)){
			glBegin(GL_LINES);
			glVertex2f(y, 0);
			glVertex2f(y, 1);
			glEnd();
		};
	}
}

float myrand(void)
{
	static int seed = 72191;
	char sq[22];

	if (ref_file)
	{
		seed *= seed;
		sprintf(sq, "%010d", seed);
		// pull the middle 5 digits out of sq
		sq[8] = 0;
		seed = atoi(&sq[3]);

		return seed / 99999.f;
	}
	else
	{
		return rand() / (float)RAND_MAX;
	}
}

void InitParticles(Vector2F *p, int dx, int dy)
{
	int i, j;
	GLint bsize;

	for (i = 0; i < dy; i++)
	{
		for (j = 0; j < dx; j++)
		{
			p[i*dx + j].x = (j + 0.5f + (myrand() - 0.5f)) / dx;
			p[i*dx + j].y = (i + 0.5f + (myrand() - 0.5f)) / dy;
		}
	}

	glGenBuffersARB(1, &vbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(Vector2F)* (DIM*DIM),
		particles, GL_DYNAMIC_DRAW_ARB);

	glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize);

	if (bsize != (sizeof(Vector2F)* DS))
		std::cout << "Error Initialising Particles" << std::endl;

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
}

void DrawParticles(float r, float g, float b, float a)
{
    glColor4f(r,g,b,a);
    glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glVertexPointer(2, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, DS);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
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