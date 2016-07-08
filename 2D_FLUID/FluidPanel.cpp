#include "cFluidPanel.h"

// Constructor
FluidPanel::FluidPanel(bool *input) : m_input_signal(input)
{}
FluidPanel::~FluidPanel()
{}

// Render Functions
void FluidPanel::Initialise(Fluid2DCPU::Parameters &parameters) 
{
	// Define GUI
	auto viscosity_scale = sfg::Scale::Create(0.f, 0.005f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto diffusion_scale = sfg::Scale::Create(0.f, 0.005f, .0001f, sfg::Scale::Orientation::HORIZONTAL);
	auto kappa_scale = sfg::Scale::Create(0.f, 1.0f, 0.1f, sfg::Scale::Orientation::HORIZONTAL);
	auto sigma_scale = sfg::Scale::Create(0.f, 1.0f, 0.1f, sfg::Scale::Orientation::HORIZONTAL);
	auto solver_scale = sfg::Scale::Create(0.f, 100.f, 10.0f, sfg::Scale::Orientation::HORIZONTAL);
	auto dt_scale = sfg::Scale::Create(0.f, 0.5f, .01f, sfg::Scale::Orientation::HORIZONTAL);
	auto vort_scale = sfg::Scale::Create(0.f, 100.f, 1.f, sfg::Scale::Orientation::HORIZONTAL);
	auto grid_check = sfg::CheckButton::Create("Show Grid");
	auto vort_check = sfg::CheckButton::Create("Vorticity");
	auto vel_check = sfg::CheckButton::Create("Velocity Field");
	auto buo_check = sfg::CheckButton::Create("Buoyancy");

	auto viscosity_label = sfg::Label::Create("Viscosity:");
	auto diffusion_label = sfg::Label::Create("Diffusion:");
	auto kappa_label = sfg::Label::Create("Kappa:");
	auto sigma_label = sfg::Label::Create("Sigma:");
	auto vort_label = sfg::Label::Create("Vorticity Strength:");
	auto solver_label = sfg::Label::Create("Solver Iteration:");
	auto dt_label = sfg::Label::Create("Time Step:");


	viscosity_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, VISCOSITY, &parameters, viscosity_scale, nullptr, viscosity_label));
	diffusion_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, DIFFUSION, &parameters, diffusion_scale, nullptr, diffusion_label));
	kappa_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, KAPPA, &parameters, kappa_scale, nullptr, kappa_label));
	sigma_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, SIGMA, &parameters, sigma_scale, nullptr, sigma_label));
	solver_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, ITERATIONS, &parameters, solver_scale, nullptr, solver_label));
	dt_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, DT, &parameters, dt_scale, nullptr, dt_label));
	vort_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, VORT_STR, &parameters, vort_scale, nullptr, vort_label));
	grid_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, GRID, &parameters, nullptr, grid_check, nullptr));
	vort_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, VORTICITY, &parameters, nullptr, vort_check, nullptr));
	buo_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, BUOYANCY, &parameters, nullptr, buo_check, nullptr));
	vel_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, VELOCITY, &parameters, nullptr, vel_check, nullptr));
	
	auto table = sfg::Table::Create();
	table->SetRowSpacings(3.f);
	table->SetColumnSpacings(3.f);

	table->Attach(sfg::Label::Create("Change the Simulation Settings."), sf::Rect<sf::Uint32>(0, 0, 3, 1), sfg::Table::FILL, sfg::Table::FILL);

	table->Attach(viscosity_label, sf::Rect<sf::Uint32>(0, 1, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(viscosity_scale, sf::Rect<sf::Uint32>(1, 1, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(diffusion_label, sf::Rect<sf::Uint32>(0, 2, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(diffusion_scale, sf::Rect<sf::Uint32>(1, 2, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(kappa_label, sf::Rect<sf::Uint32>(0, 3, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(kappa_scale, sf::Rect<sf::Uint32>(1, 3, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(sigma_label, sf::Rect<sf::Uint32>(0, 4, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(sigma_scale, sf::Rect<sf::Uint32>(1, 4, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(vort_label, sf::Rect<sf::Uint32>(0, 5, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(vort_scale, sf::Rect<sf::Uint32>(1, 5, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(solver_label, sf::Rect<sf::Uint32>(0, 6, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(solver_scale, sf::Rect<sf::Uint32>(1, 6, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);

	table->Attach(dt_label, sf::Rect<sf::Uint32>(0, 7, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(dt_scale, sf::Rect<sf::Uint32>(1, 7, 1, 1), sfg::Table::FILL | sfg::Table::EXPAND, sfg::Table::FILL | sfg::Table::EXPAND);
	table->Attach(vort_check, sf::Rect<sf::Uint32>(1, 8, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(buo_check, sf::Rect<sf::Uint32>(1, 9, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(grid_check, sf::Rect<sf::Uint32>(1, 10, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(vel_check, sf::Rect<sf::Uint32>(1, 11, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	
	auto window = sfg::Window::Create();
	window->SetTitle("Fluid Panel");
	window->SetPosition(sf::Vector2f(50.f, 100.f));
	window->GetSignal(sfg::Window::OnMouseEnter).Connect(std::bind(&FluidPanel::OnMouseEnter, this, m_input_signal));
	window->GetSignal(sfg::Window::OnMouseLeave).Connect(std::bind(&FluidPanel::OnMouseLeave, this, m_input_signal));
	window->Add(table);
	window->SetStyle(window->GetStyle() ^ sfg::Window::BACKGROUND);
	
	desktop.SetProperty("Window", "TitleBackgroundColor", sf::Color::Blue); // for all labels
	desktop.Add(window);

	viscosity_scale->SetValue(parameters.viscosity);
	diffusion_scale->SetValue(parameters.diffusion);
	kappa_scale->SetValue(parameters.kappa);
	sigma_scale->SetValue(parameters.sigma);
	solver_scale->SetValue(parameters.iterations);
	dt_scale->SetValue(parameters.dt);
	vort_scale->SetValue(parameters.vort_str);
	buo_check->SetActive(parameters.buoyancy);
	vort_check->SetActive(parameters.vorticity);
	vel_check->SetActive(parameters.velocity);

	viscosity_label->SetText(PrintText("Viscosity", parameters.viscosity,4));
	diffusion_label->SetText(PrintText("Diffusion", parameters.diffusion,4));
	kappa_label->SetText(PrintText("Kappa", parameters.kappa,2));
	sigma_label->SetText(PrintText("Sigma", parameters.sigma,2));
	vort_label->SetText(PrintText("Vorticity Strength", parameters.vort_str,0));
	solver_label->SetText(PrintText("Solver Iteration", parameters.iterations,0));
	dt_label->SetText(PrintText("Time Step", parameters.dt,2));
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
void FluidPanel::OnScaleChange(PARMAP param_map, 
							   Fluid2DCPU::Parameters *parameters, 
							   std::shared_ptr<sfg::Scale> scale_ptr, 
							   std::shared_ptr<sfg::CheckButton> check_ptr,
							   std::shared_ptr<sfg::Label> label_ptr)
{
	switch (param_map)
	{
	case ITERATIONS:
		parameters->iterations = scale_ptr->GetValue();
		label_ptr->SetText(PrintText("Solver Iteration", parameters->iterations, 0));
		break;
	case DIFFUSION:
		label_ptr->SetText(PrintText("Diffusion", parameters->diffusion, 4));
		parameters->diffusion = scale_ptr->GetValue();
		break;
	case VISCOSITY:
		label_ptr->SetText(PrintText("Viscosity", parameters->viscosity, 4));
		parameters->viscosity = scale_ptr->GetValue();
		break;
	case KAPPA:
		label_ptr->SetText(PrintText("Kappa", parameters->kappa, 2));
		parameters->kappa = scale_ptr->GetValue();
		break;
	case SIGMA:
		label_ptr->SetText(PrintText("Sigma", parameters->sigma, 2));
		parameters->sigma = scale_ptr->GetValue();
		break;
	case VORT_STR:
		label_ptr->SetText(PrintText("Vorticity Strength", parameters->vort_str, 0));
		parameters->vort_str = scale_ptr->GetValue();
		break;
	case DT:
		label_ptr->SetText(PrintText("Time Step", parameters->dt, 2));
		parameters->dt = scale_ptr->GetValue();
		break;
	//	Boolean Check
	case GRID:
		parameters->grid = check_ptr->IsActive();
		break;
	case BUOYANCY:
		parameters->buoyancy = check_ptr->IsActive();
		break;
	case VORTICITY:
		parameters->vorticity = check_ptr->IsActive();
		break;
	case VELOCITY:
		parameters->velocity = check_ptr->IsActive();
		break;
	default:
		break;
	}
}

void FluidPanel::OnMouseEnter(bool *input)
{
	*input = true;
}

void FluidPanel::OnMouseLeave(bool *input)
{
	*input = false;
}