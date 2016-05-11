#include "FluidPanel.h"

// Constructor
FluidPanel::FluidPanel(bool *input) : m_input_signal(input)
{}
FluidPanel::~FluidPanel()
{}

// Render Functions
void FluidPanel::Initialise()
{
	m_parameters.iterations = 10;
	m_parameters.dt = 0.01f;
	m_parameters.kappa = 0.3f;
	m_parameters.sigma = 0.f;
	m_parameters.diffusion = 0.f;
	m_parameters.viscosity = 0.f;
	m_parameters.vort_str = 10.f;
	m_parameters.vorticity = false;
	m_parameters.buoyancy = true;
	m_parameters.grid = false;
	m_parameters.isMaccormack = true;
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
	auto buo_check = sfg::CheckButton::Create("Buoyancy");
	auto mac_check = sfg::CheckButton::Create("MacCormack");

	auto viscosity_label = sfg::Label::Create("Viscosity:");
	auto diffusion_label = sfg::Label::Create("Diffusion:");
	auto kappa_label = sfg::Label::Create("Kappa:");
	auto sigma_label = sfg::Label::Create("Sigma:");
	auto vort_label = sfg::Label::Create("Vorticity Str:");
	auto solver_label = sfg::Label::Create("Solver Iter:");
	auto dt_label = sfg::Label::Create("Time Step:");


	viscosity_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, VISCOSITY, &m_parameters, viscosity_scale, nullptr, viscosity_label));
	diffusion_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, DIFFUSION, &m_parameters, diffusion_scale, nullptr, diffusion_label));
	kappa_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, KAPPA, &m_parameters, kappa_scale, nullptr, kappa_label));
	sigma_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, SIGMA, &m_parameters, sigma_scale, nullptr, sigma_label));
	solver_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, ITERATIONS, &m_parameters, solver_scale, nullptr, solver_label));
	dt_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, DT, &m_parameters, dt_scale, nullptr, dt_label));
	vort_scale->GetSignal(sfg::Scale::OnMouseMove).Connect(std::bind(&FluidPanel::OnScaleChange, this, VORT_STR, &m_parameters, vort_scale, nullptr, vort_label));
	grid_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, GRID, &m_parameters, nullptr, grid_check, nullptr));
	vort_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, VORTICITY, &m_parameters, nullptr, vort_check, nullptr));
	buo_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, BUOYANCY, &m_parameters, nullptr, buo_check, nullptr));
	mac_check->GetSignal(sfg::CheckButton::OnToggle).Connect(std::bind(&FluidPanel::OnScaleChange, this, MACCORMACK, &m_parameters, nullptr, mac_check, nullptr));

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
	table->Attach(mac_check, sf::Rect<sf::Uint32>(1, 10, 1, 1), sfg::Table::FILL, sfg::Table::FILL);
	table->Attach(grid_check, sf::Rect<sf::Uint32>(1, 11, 1, 1), sfg::Table::FILL, sfg::Table::FILL);

	auto window = sfg::Window::Create();
	window->SetTitle("Fluid Panel");
	window->SetPosition(sf::Vector2f(0.f, 100.f));
	window->GetSignal(sfg::Window::OnMouseEnter).Connect(std::bind(&FluidPanel::OnMouseEnter, this, m_input_signal));
	window->GetSignal(sfg::Window::OnMouseLeave).Connect(std::bind(&FluidPanel::OnMouseLeave, this, m_input_signal));
	window->Add(table);
	window->SetStyle(window->GetStyle() ^ sfg::Window::BACKGROUND);

	desktop.SetProperty("Window", "TitleBackgroundColor", sf::Color::Green);
	desktop.SetProperty("Window", "Color", sf::Color::Black);
	desktop.Add(window);

	viscosity_scale->SetValue(m_parameters.viscosity);
	diffusion_scale->SetValue(m_parameters.diffusion);
	kappa_scale->SetValue(m_parameters.kappa);
	sigma_scale->SetValue(m_parameters.sigma);
	solver_scale->SetValue(m_parameters.iterations);
	dt_scale->SetValue(m_parameters.dt);
	vort_scale->SetValue(m_parameters.vort_str);
	buo_check->SetActive(m_parameters.buoyancy);
	vort_check->SetActive(m_parameters.vorticity);
	mac_check->SetActive(m_parameters.isMaccormack);

	viscosity_label->SetText(PrintText("Viscosity", m_parameters.viscosity, 4));
	diffusion_label->SetText(PrintText("Diffusion", m_parameters.diffusion, 4));
	kappa_label->SetText(PrintText("Kappa", m_parameters.kappa, 2));
	sigma_label->SetText(PrintText("Sigma", m_parameters.sigma, 2));
	vort_label->SetText(PrintText("Vorticity Str", m_parameters.vort_str, 0));
	solver_label->SetText(PrintText("Solver Iter", m_parameters.iterations, 0));
	dt_label->SetText(PrintText("Time Step", m_parameters.dt, 2));
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
	Parameters *m_parameters,
	std::shared_ptr<sfg::Scale> scale_ptr,
	std::shared_ptr<sfg::CheckButton> check_ptr,
	std::shared_ptr<sfg::Label> label_ptr)
{
	switch (param_map)
	{
	case ITERATIONS:
		m_parameters->iterations = scale_ptr->GetValue();
		label_ptr->SetText(PrintText("Solver Iter", m_parameters->iterations, 0));
		break;
	case DIFFUSION:
		label_ptr->SetText(PrintText("Diffusion", m_parameters->diffusion, 4));
		m_parameters->diffusion = scale_ptr->GetValue();
		break;
	case VISCOSITY:
		label_ptr->SetText(PrintText("Viscosity", m_parameters->viscosity, 4));
		m_parameters->viscosity = scale_ptr->GetValue();
		break;
	case KAPPA:
		label_ptr->SetText(PrintText("Kappa", m_parameters->kappa, 2));
		m_parameters->kappa = scale_ptr->GetValue();
		break;
	case SIGMA:
		label_ptr->SetText(PrintText("Sigma", m_parameters->sigma, 2));
		m_parameters->sigma = scale_ptr->GetValue();
		break;
	case VORT_STR:
		label_ptr->SetText(PrintText("Vorticity Str", m_parameters->vort_str, 0));
		m_parameters->vort_str = scale_ptr->GetValue();
		break;
	case DT:
		label_ptr->SetText(PrintText("Time Step", m_parameters->dt, 2));
		m_parameters->dt = scale_ptr->GetValue();
		break;
		//	Boolean Check
	case GRID:
		m_parameters->grid = check_ptr->IsActive();
		break;
	case BUOYANCY:
		m_parameters->buoyancy = check_ptr->IsActive();
		break;
	case VORTICITY:
		m_parameters->vorticity = check_ptr->IsActive();
		break;
	case MACCORMACK:
		m_parameters->isMaccormack = check_ptr->IsActive();
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