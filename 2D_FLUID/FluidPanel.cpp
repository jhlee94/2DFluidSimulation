#include "cFluidPanel.h"

// Constructor
FluidPanel::FluidPanel(bool &input) : m_input_signal(input)
{}
FluidPanel::~FluidPanel()
{}

// Render Functions
void FluidPanel::Initialise(Fluid2DCPU::Parameters &parameters) 
{
	// Define GUI
	auto viscosity_scale = sfg::Scale::Create(0.f, 0.001f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto diffusion_scale = sfg::Scale::Create(0.f, 0.001f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto kappa_scale = sfg::Scale::Create(0.f, 1.0f, 0.1f, sfg::Scale::Orientation::HORIZONTAL);
	auto sigma_scale = sfg::Scale::Create(0.f, 1.0f, 0.1f, sfg::Scale::Orientation::HORIZONTAL);
	auto solver_scale = sfg::Scale::Create(0.f, 100.f, 10.0f, sfg::Scale::Orientation::HORIZONTAL);
	auto dt_scale = sfg::Scale::Create(0.f, 0.5f, .01f, sfg::Scale::Orientation::HORIZONTAL);
	auto grid_check = sfg::CheckButton::Create("Show Grid");

	viscosity_scale->GetSignal(sfg::Scale::OnLeftClick).Connect(std::bind(&FluidPanel::OnScaleChange, this, VISCOSITY, &parameters, viscosity_scale));
	diffusion_scale->GetSignal(sfg::Scale::OnLeftClick).Connect(std::bind(&FluidPanel::OnScaleChange, this, DIFFUSION, &parameters, diffusion_scale));
	kappa_scale->GetSignal(sfg::Scale::OnLeftClick).Connect(std::bind(&FluidPanel::OnScaleChange, this, KAPPA, &parameters, kappa_scale));
	sigma_scale->GetSignal(sfg::Scale::OnLeftClick).Connect(std::bind(&FluidPanel::OnScaleChange, this, SIGMA, &parameters, sigma_scale));
	solver_scale->GetSignal(sfg::Scale::OnLeftClick).Connect(std::bind(&FluidPanel::OnScaleChange, this, ITERATIONS, &parameters, solver_scale));
	dt_scale->GetSignal(sfg::Scale::OnLeftClick).Connect(std::bind(&FluidPanel::OnScaleChange, this, DT, &parameters, dt_scale));

	auto table = sfg::Table::Create();
	table->SetRowSpacings(5.f);
	table->SetColumnSpacings(5.f);

	table->Attach(sfg::Label::Create("Change the Simulation Settings."), sf::Rect<sf::Uint32>(0, 0, 3, 1), sfg::Table::FILL, sfg::Table::FILL);

	table->Attach(sfg::Label::Create("Viscosity:"), sf::Rect<sf::Uint32>(0, 1, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(viscosity_scale, sf::Rect<sf::Uint32>(1, 1, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(sfg::Label::Create("Diffusion:"), sf::Rect<sf::Uint32>(0, 2, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(diffusion_scale, sf::Rect<sf::Uint32>(1, 2, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(sfg::Label::Create("Kappa:"), sf::Rect<sf::Uint32>(0, 3, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(kappa_scale, sf::Rect<sf::Uint32>(1, 3, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(sfg::Label::Create("Sigma:"), sf::Rect<sf::Uint32>(0, 4, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(sigma_scale, sf::Rect<sf::Uint32>(1, 4, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(sfg::Label::Create("Solver Iteration:"), sf::Rect<sf::Uint32>(0, 5, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(solver_scale, sf::Rect<sf::Uint32>(1, 5, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(sfg::Label::Create("Time Step:"), sf::Rect<sf::Uint32>(0, 6, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(dt_scale, sf::Rect<sf::Uint32>(1, 6, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(grid_check, sf::Rect<sf::Uint32>(1, 7, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	
	auto window = sfg::Window::Create();
	window->SetTitle("Fluid Panel");
	window->SetPosition(sf::Vector2f(100.f, 100.f));
	window->GetSignal(sfg::Window::OnMouseEnter).Connect(std::bind(&FluidPanel::GUICatch, this, m_input_signal));
	window->GetSignal(sfg::Window::OnMouseLeave).Connect(std::bind(&FluidPanel::GUICatch, this, m_input_signal));
	window->Add(table);
	
	desktop.Add(window);

	viscosity_scale->SetValue(parameters.viscosity);
	diffusion_scale->SetValue(parameters.diffusion);
	kappa_scale->SetValue(parameters.kappa);
	sigma_scale->SetValue(parameters.sigma);
	solver_scale->SetValue(parameters.iterations);
	dt_scale->SetValue(parameters.dt);
}
void FluidPanel::Update(float dt)
{
	desktop.Update(dt);
}
void FluidPanel::Display(sf::RenderWindow& window)
{
	m_sfgui.Display(window);
}


void FluidPanel::HandleEvent(sf::Event &event)
{
	desktop.HandleEvent(event);
}

// Scale Button Functions
void FluidPanel::OnScaleChange(PARMAP param_map, Fluid2DCPU::Parameters *parameters, std::shared_ptr<sfg::Scale> pointer)
{
	switch (param_map)
	{
	case ITERATIONS:
		parameters->iterations = pointer->GetValue();
		break;
	case DIFFUSION:
		parameters->diffusion = pointer->GetValue();
		break;
	case VISCOSITY:
		parameters->viscosity = pointer->GetValue();
		break;
	case KAPPA:
		parameters->kappa = pointer->GetValue();
		break;
	case SIGMA:
		parameters->sigma = pointer->GetValue();
		break;
	case DT:
		parameters->dt = pointer->GetValue();
		break;
	default:
		break;
	}
}

void FluidPanel::GUICatch(bool &input)
{
	input = !input;
	std::cout << input << std::endl;
}