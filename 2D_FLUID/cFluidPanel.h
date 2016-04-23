#ifndef _FLUID_PANEL_H_
#define _FLUID_PANEL_H_

// Includes
#include <SFML\Graphics\RenderWindow.hpp>
#include <SFGUI\SFGUI.hpp>
#include <SFGUI\Widgets.hpp>
#include "Fluid.h"

class FluidPanel
{
private:
	enum PARMAP
	{
		ITERATIONS,
		VISCOSITY,
		DIFFUSION,
		KAPPA,
		SIGMA,
		DT
	};

	sfg::SFGUI m_sfgui;
	sfg::Desktop desktop;
	bool &m_input_signal;

public:
	// Constructor
	FluidPanel(bool &input);
	virtual ~FluidPanel();

	// Render Functions
	void Initialise(Fluid2DCPU::Parameters &parameters);
	void Update(float dt);
	void Display(sf::RenderWindow& window);
	void HandleEvent(sf::Event &event);

	// Scale Button Functions
	void OnScaleChange(PARMAP param_map, Fluid2DCPU::Parameters *parameters, std::shared_ptr<sfg::Scale> pointer);

	// GUI Input Catcher
	void GUICatch(bool &input);
};

#endif // !_FLUID_PANEL_H_
