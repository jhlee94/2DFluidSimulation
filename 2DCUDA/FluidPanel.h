#ifndef _FLUID_PANEL_H_
#define _FLUID_PANEL_H_

// Includes
#include <sstream>
#include <SFML\Graphics\RenderWindow.hpp>
#include <SFGUI\SFGUI.hpp>
#include <SFGUI\Widgets.hpp>

#include "config.h"

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
		DT,
		VORT_STR,
		VORTICITY,
		BUOYANCY,
		GRID,
		MACCORMACK
	};

	sfg::SFGUI m_sfgui;
	sfg::Desktop desktop;
	bool *m_input_signal;


	inline std::string PrintText(std::string text, const float value, int precision)
	{
		std::ostringstream oss;
		oss << text << " [" << std::setprecision(precision) << std::fixed << value << "]";
		return oss.str();
	}

public:
	Parameters m_parameters;
	
public:
	// Constructor
	FluidPanel(bool *input);
	virtual ~FluidPanel();

	// Render Functions
	void Initialise();
	void Update(float dt);
	void Display(sf::RenderWindow& window);
	void HandleEvent(sf::Event &event);

	// Scale Button Functions
	void OnScaleChange(PARMAP param_map,
		Parameters *parameters,
		std::shared_ptr<sfg::Scale> pointer,
		std::shared_ptr<sfg::CheckButton> check_ptr,
		std::shared_ptr<sfg::Label> label_ptr);

	// GUI Input Catcher
	void OnMouseLeave(bool *input);
	void OnMouseEnter(bool *input);
};

#endif // !_FLUID_PANEL_H_
