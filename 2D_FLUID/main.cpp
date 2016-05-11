#pragma warning (disable:4996)
#include <ctime>
#include <SFGUI\SFGUI.hpp>
#include <SFGUI/Widgets.hpp>

#include <GL\glew.h>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>

#include "cFluidPanel.h"
#define CLAMP(v, a, b) (a + (v - a) / (b - a))

sf::Clock fps_clock, sim_clock;
float current_time, previous_time = 0.f, frame_count = 0.f, fps = 0.f;
sf::Font* main_font;

sf::Texture texture;
sf::Sprite* fluid_sprite;

int mouseX0 = 0, mouseY0 = 0;

// Fluid GUI Panel
FluidPanel* panel = { nullptr };
bool gui = false; // GUI input signal


// FluidSolver
Fluid2DCPU* fluid_solver;
Vector2F* particles = NULL;


sf::CircleShape circle;

void Init();
void InitGL();
void Clean();
void Display(sf::RenderWindow &window);
void DrawGrid(bool);
void DrawVectorField(bool);
void CreateGUI(sfg::Desktop& desktop);
void PrintString(float x, float y, sf::Text& text, const char* string, ...);
void CalculateFPS(void);
void ApplyColour(float x, float, float);
void HandleInput(sf::RenderWindow &window, sf::Event &event);

int main()
{
	// An sf::Window for raw OpenGL rendering.
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::RenderWindow app_window(sf::VideoMode(WIDTH, HEIGHT), "2D Fluid Simulator CPU", sf::Style::Close|sf::Style::Titlebar, settings);
	//app_window.setVerticalSyncEnabled(true);

	// Init

	Init();
	app_window.setActive();
	InitGL();

	circle.setRadius(10);
	circle.setPosition(fluid_sprite->getPosition());
	circle.setFillColor(sf::Color::White);

	// SFML mainloop
	sf::Event event;
	while (app_window.isOpen()) {
		CalculateFPS();
		// Handle Input
		HandleInput(app_window, event);
		// Draw
		Display(app_window);
	}

	Clean();
	
	return 0;
}

void Clean()
{
	// cleanup
	glDisable(GL_LINE_SMOOTH);
	delete main_font;
	delete fluid_sprite;
	delete fluid_solver;
	delete panel;
}

void Init()
{
	main_font = new sf::Font;
	main_font->loadFromFile("../Resources/arial.ttf");

	fluid_solver = new Fluid2DCPU();
	fluid_solver->Initialise(DIM);

	// create an empty 200x200 texture
	if (!texture.create((DIM+2), (DIM+2)))
	{
		// error...
	}
	texture.setSmooth(true);

	fluid_sprite = new sf::Sprite(texture);
	fluid_sprite->setOrigin((DIM + 2) / 2.f, (DIM + 2) / 2.f);
	fluid_sprite->setPosition(WIDTH / 2 + 100.f, HEIGHT / 2);
	fluid_sprite->setScale((GRID_WIDTH+2.f)/(DIM+2.f),(GRID_HEIGHT+2.f)/(DIM+2.f));

	panel = new FluidPanel(&gui);
	panel->Initialise(fluid_solver->m_parameters);
}

void InitGL()
{
	// Init GLEW functions
	glewInit();
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	// GL_Display Init
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Display(sf::RenderWindow &window)
{
	sf::Text fps_text;
	auto delta = sim_clock.restart().asSeconds();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	
	// Step Fluid
	//fluid_solver->m_parameters.dt = delta;
	fluid_solver->step();
	texture.update(fluid_solver->pixels);

	//// Render Density
	//glPushMatrix();
	//glTranslatef((WIDTH/2.f - GRID_WIDTH/2.f) + 100.f, (HEIGHT/2.f - GRID_HEIGHT/2.f), 0.f);
	//glColor3f(1, 1, 1);
	//glBegin(GL_LINE_LOOP);
	//glLineWidth(0.1f);
	//glVertex2i(-1, -1);
	//glVertex2i(GRID_WIDTH+1, -1);
	//glVertex2i(GRID_WIDTH+1, GRID_HEIGHT+1);
	//glVertex2i(-1, GRID_HEIGHT+1);
	//glEnd();
	
	//// Grid Lines 
	//glPopMatrix();

	

	// SFML rendering.
	// Draw FPS Text
	window.pushGLStates();
	window.draw(*fluid_sprite);
	//window.draw(circle);

	PrintString(5, 16, fps_text, "FPS: %5.2f", fps);
	window.draw(fps_text);

	// SFGUI Update
	panel->Update(delta);
	panel->Display(window);
	window.popGLStates();

	DrawGrid(fluid_solver->m_parameters.grid);
	DrawVectorField(fluid_solver->m_parameters.velocity);
	// Finally, Display all
	window.display();
	//glFlush();
}

void DrawGrid(bool x)
{
	if (x)
	{
		glPushMatrix();
		glTranslatef(fluid_sprite->getPosition().x - GRID_WIDTH / 2.f, fluid_sprite->getPosition().y - GRID_HEIGHT / 2.f, 0.f);
		glColor4f(1.f, 1.f, 1.f, 1.0f);
		for (float x = TILE_SIZE_X; x < GRID_WIDTH; x += TILE_SIZE_X){
			glBegin(GL_LINES);
			glVertex2f(0, x);
			glVertex2f(GRID_WIDTH, x);
			glEnd();
		};
		for (float y = TILE_SIZE_Y; y < GRID_HEIGHT; y += TILE_SIZE_Y){
			glBegin(GL_LINES);
			glVertex2f(y, 0);
			glVertex2f(y, GRID_HEIGHT);
			glEnd();
		};
		glPopMatrix();
	}
}


void DrawVectorField(bool c)
{
	if (c){
		float dx, dy;
		for (int i = 0; i < DS; i++) {
			int x = i % (DIM + 2);
			int y = i / (DIM + 2);
			dx = (x)* TILE_SIZE_X;
			dy = (y)* TILE_SIZE_Y;
			int cell_idx = fluid_solver->index(x, y);
			float dx_u = dx + fluid_solver->u[cell_idx] * 10.f;
			float dy_v = dy + fluid_solver->v[cell_idx] * 10.f;
			glPushMatrix();
			glTranslatef(fluid_sprite->getPosition().x - GRID_WIDTH / 2.f, fluid_sprite->getPosition().y - GRID_HEIGHT / 2.f, 0.f);
			//glScalef(0.5f, 0.5f, 1.0);
			glBegin(GL_LINES);
			glColor3f(255, 255, 255);
			glVertex2f(dx, dy);
			glVertex2f(dx_u, dy_v);
			glEnd();
			glPopMatrix();
		}
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

void HandleInput(sf::RenderWindow &window, sf::Event &event)
{
	while (window.pollEvent(event)) {

		panel->HandleEvent(event);
		if (gui) break;
		switch (event.type)
		{
		case sf::Event::Closed:
			window.close();
			break;
		case sf::Event::Resized:
			glViewport(0, 0, event.size.width, event.size.height);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, WIDTH, HEIGHT, 0, 0, 1);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
		case sf::Event::LostFocus:
			break;
		case sf::Event::MouseButtonPressed:
		{
			int i = (event.mouseButton.x / static_cast<float>(WIDTH)) * DIM + 1;
			int j = (event.mouseButton.y / static_cast<float>(HEIGHT)) * DIM + 1;
			break;
		}
		case sf::Event::MouseMoved:
		{
			int mouseX = event.mouseMove.x;
			int mouseY = event.mouseMove.y;
			if ((mouseX >= 0 && mouseX < WIDTH) && (mouseY >= 0 && mouseY < HEIGHT)){
				int i = (mouseX / static_cast<float>(WIDTH)) * DIM + 1;
				int j = (mouseY / static_cast<float>(HEIGHT)) * DIM + 1;
				float dirX = (mouseX - mouseX0) * 100;
				float dirY = (mouseY - mouseY0) * 100;
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
			break;
		}
		default:
			break;
		}
		
	}

	if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && !gui){
		int i = (sf::Mouse::getPosition(window).x / static_cast<float>(WIDTH)) * DIM + 1;
		int j = (sf::Mouse::getPosition(window).y / static_cast<float>(HEIGHT)) * DIM + 1;
		fluid_solver->dens[fluid_solver->index(i, j)] = 10.f;
		fluid_solver->dens[fluid_solver->index(i - 1, j)] = 10.f;
		fluid_solver->dens[fluid_solver->index(i + 1, j)] = 10.f;
		fluid_solver->dens[fluid_solver->index(i - 2, j)] = 10.f;
		fluid_solver->dens[fluid_solver->index(i + 2, j)] = 10.f;
	}
}