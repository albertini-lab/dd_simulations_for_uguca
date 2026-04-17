/**
 * @file   dd_earthquake.cc
 * @brief  Data-driven simulation for laboratory generated earthquakes.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <optional>   
#include <stdexcept>  

#include "static_communicator_mpi.hh"
#include "uca_parameter_reader.hh"
#include "uca_simple_mesh.hh"
#include "material.hh"
#include "nodal_field.hh"
#include "unimat_shear_interface.hh"
#include "uca_restart.hh"

#include "rate_and_state_law.hh"
#include "slip_weakening_friction_law.hh"
#include "linear_shear_cohesive_law.hh"

namespace fs = std::filesystem;
using namespace uguca;

int main(int argc, char* argv[]) {
  StaticCommunicatorMPI* comm = StaticCommunicatorMPI::getInstance();
  int world_rank = comm->whoAmI();

  if (argc < 2) {
    if (world_rank == 0) std::cerr << "Usage: ./dd_earthquake <input_file>\n";
    return 1;
  }
  
  if (world_rank == 0) std::cout << "\n========== [DEBUG] UGUCA DD INITIALISATION ==========\n";
  
  ParameterReader data;
  data.readInputFile(argv[1]);

  double a=0.0, b=0.0, Dc=0.0, V0=0.0, f0=0.0, V_init=0.0;
  
  std::string simulation_name = data.get<std::string>("simulation_name");
  std::string dump_folder = data.get<std::string>("dump_folder");
  if (!dump_folder.empty() && dump_folder.back() != '/') dump_folder += "/";

  double length_x = data.get<double>("length");
  int nb_elements = data.get<int>("nb_elements");
  bool free_surface = data.get<bool>("free_surface");
  SimpleMesh mesh(length_x, nb_elements);

  double duration = data.get<double>("duration");
  double time_step_factor = data.get<double>("time_step_factor");
  int nb_pc = data.get<int>("nb_predictor_corrector");
  double dump_interval = data.getOrUse<double>("dump_interval", -1.0);

  double E = data.get<double>("E");
  double nu = data.get<double>("nu");
  double rho = data.get<double>("rho");
  Material mat = Material(E, nu, rho);
  mat.readPrecomputedKernels();

  double shear_load = data.get<double>("shear_load");
  double normal_load = data.get<double>("normal_load");

  double nuc_center = data.get<double>("nuc_center");
  double delta_tau_0 = data.get<double>("nuc_dtau");
  double R = data.get<double>("nuc_size");
  double T = data.get<double>("nuc_time");

  std::string law_name = data.get<std::string>("Law");
  InterfaceLaw* law = nullptr;

  if (law_name == "RateAndState") {
      a = data.get<double>("a"); b = data.get<double>("b"); Dc = data.get<double>("Dc");
      V0 = data.get<double>("V0"); f0 = data.get<double>("f0"); V_init = data.get<double>("V_init");

      double theta_init = Dc / V0 * std::exp((a * std::log(2 * std::sinh(shear_load / a / std::abs(normal_load))) - f0 - a * std::log(V_init / V0)) / b);
      std::string ev_str = data.get<std::string>("evolution_law");
      RateAndStateLaw::EvolutionLaw ev_law = (ev_str == "Slip") ? RateAndStateLaw::EvolutionLaw::SlipLaw : RateAndStateLaw::EvolutionLaw::AgingLaw;
      law = new RateAndStateLaw(mesh, a, b, Dc, V0, f0, theta_init, ev_law, nb_pc > 0, 0.0);
  }

  UnimatShearInterface interface(mesh, {_x,_y}, mat, *law);
  interface.getLoad().setAllValuesTo(shear_load, _x);
  interface.getLoad().setAllValuesTo(normal_load, _y);

  if (law_name == "RateAndState") {
      interface.getTop().getVelo().setAllValuesTo(V_init / 2, _x);
      if (data.get<bool>("asperity")) {
          RateAndStateLaw* rs_law = dynamic_cast<RateAndStateLaw*>(law);
          double asp_center = data.get<double>("asp_center");
          double fct = data.get<double>("asp_fct");
          double R_asp = data.get<double>("asp_size");
          bool is_box_car = data.getOrUse<bool>("box_car_shape", false);
          const TwoDVector& coords = mesh.getLocalCoords();
          for (int i = 0; i < mesh.getNbLocalNodes(); ++i) {
              double x = std::abs(coords(i, _x) - asp_center);
              double r = std::sqrt(x * x);
              double perturb = 0.0;
              if (is_box_car) { if (r < R_asp) perturb = 1.0; } 
              else { if (r < R_asp) perturb = std::exp(r * r / (r * r - R_asp * R_asp)); }
              double normal_load_bump = normal_load + normal_load * (fct - 1) * perturb;
              double shear_load_bump = normal_load_bump / normal_load * shear_load;
              interface.getLoad()(i, _y) = normal_load_bump;
              interface.getLoad()(i, _x) = shear_load_bump;
              rs_law->getTheta()(i) = Dc / V0 * std::exp((a * std::log(2 * std::sinh(shear_load_bump / a / std::abs(normal_load_bump))) - f0 - a * std::log(V_init / V0)) / b);
          }
      }
  }

  double time_step = time_step_factor * interface.getStableTimeStep();
  interface.setTimeStep(time_step);
  interface.initPredictorCorrector(nb_pc);
  interface.init();

  NodalField shear_load_init(mesh, {_x,_y});
  for (int i = 0; i < mesh.getNbLocalNodes(); ++i) shear_load_init(i, _x) = interface.getLoad()(i, _x);

  bool dd_active = data.getOrUse<bool>("enable_data_driven_mode", false);
  NodalField top_u_data(mesh, {_x, _y}), top_w_f(mesh, {_x, _y}), loaded_weights(mesh, {_x, _y});
  std::optional<Restart> loader;
  double w_f_val = 0.0, n_var = 0.0;
  std::string gt_path = "", gt_name = "";

  if (dd_active) {
      w_f_val = data.get<double>("w_factor_constant");
      n_var = std::pow(data.get<double>("noise_stddev_sim"), 2);
      gt_path = data.get<std::string>("ground_truth_data_path");
      gt_name = data.get<std::string>("ground_truth_data_name");

      try {
          loader.emplace(gt_name, gt_path, Restart::Format::Binary);
          loader->registerIO("top_disp", top_u_data); 
          loader->registerIO("top_w_factor", loaded_weights);
      } catch (...) {
          dd_active = false;
      }
  }

  int nb_steps = std::ceil(duration / time_step);
  int s_dump = (dump_interval > 0.0) ? std::ceil(dump_interval / time_step) : std::max(1, nb_steps / 100);
  
  if (world_rank == 0) {
      std::cout << "[DEBUG] Total Simulation Steps = " << nb_steps << "\n";
      std::cout << "\n========== [DEBUG] EXECUTING MAIN LOOP ==========\n";
  }

  fs::create_directories(dump_folder);
  interface.initDump(simulation_name + "-interface", dump_folder, Dumper::Format::ASCII);
  std::stringstream ss(data.get<std::string>("dump_fields"));
  std::string field;
  while (std::getline(ss, field, ',')) interface.registerDumpField(field);
  interface.dump(0, 0);

  const TwoDVector& coords = mesh.getLocalCoords();

  for (int s = 1; s <= nb_steps; ++s) {
    double t = time_step * s;
    
    for (int i = 0; i < mesh.getNbLocalNodes(); ++i) {
            double x = std::abs(coords(i, _x) - nuc_center);
            double r = std::sqrt(x * x);
      double perturb = 0.0;
      if (r < R) perturb = std::exp(r * r / (r * r - R * R));
      double G = 1.0;
      if (t < T) G = std::exp((t - T) * (t - T) / t / (t - 2.0 * T));
      interface.getLoad()(i, _x) = shear_load_init(i, _x) + delta_tau_0 * perturb * G;
    }

    if (free_surface) {
      int nb_nodes_x = mesh.getNbLocalNodes();
      for (int i = 0; i < nb_nodes_x / 2; ++i) {
        int ijsym = nb_nodes_x - i - 1;
        interface.getTop().getDisp()(ijsym, _x) = -interface.getTop().getDisp()(i, _x);
        interface.getTop().getVelo()(ijsym, _x) = -interface.getTop().getVelo()(i, _x);
      }
    }

    bool physics_only = true;
    if (dd_active && s % data.getOrUse<int>("data_driven_update_interval", 1) == 0) {
        
        std::string expected_file = gt_path + "/" + gt_name + "-restart/top_w_factor.proc0.s" + std::to_string(s) + ".out";
        
        if (fs::exists(expected_file)) {
            try {
                loader->load(s);
                physics_only = false;
                
                double active_nodes = 0.0;
                for (int i = 0; i < loaded_weights.getNbNodes(); ++i) {
                    top_w_f(i, _x) = loaded_weights(i, _x) * w_f_val;
                    top_w_f(i, _y) = loaded_weights(i, _y) * w_f_val;
                    if (loaded_weights(i, _x) > 0.0) active_nodes++;
                }
                
                if (world_rank == 0) {
                    std::cout << "\n[========================================================]\n";
                    std::cout << "[SUCCESS-DD] Step " << s << "/" << nb_steps << " | Data File Found and Loaded!\n";
                    std::cout << "[SUCCESS-DD] Applying w=" << w_f_val << " to " << active_nodes << " active nodes.\n";
                    std::cout << "[========================================================]\n\n";
                }

                interface.getTop().computeDisplacementWithData(top_u_data, top_w_f, n_var, false, 1);
                interface.computeInternal(false, false, _dynamic, 1);
                interface.computeCohesion(false, 1);
                interface.computeResidual();
                interface.computeVelocity();
            } catch (...) { 
                physics_only = true; 
            }
        } else {
            physics_only = true;
        }
    } 
    
    if (physics_only) interface.advanceTimeStep();

    if (s % s_dump == 0) {
        interface.dump(s, t);
        if (world_rank == 0) {
            double max_u = 0.0, max_v = 0.0;
            for (int i = 0; i < mesh.getNbLocalNodes(); ++i) {
                max_u = std::max(max_u, std::abs(interface.getTop().getDisp()(i, _x)));
                max_v = std::max(max_v, std::abs(interface.getTop().getVelo()(i, _x)));
            }
            std::cout << "[DEBUG-PHYS] Step " << s << "/" << nb_steps << " | max(|u|) = " << max_u << " m | max(|v|) = " << max_v << " m/s | PC=" << nb_pc << "\n";
        }
    }
  }
  
  if (world_rank == 0) std::cout << "\n========== [DEBUG] RUN COMPLETED SUCCESSFULLY ==========\n";
  StaticCommunicatorMPI::getInstance()->finalize();
  return 0;
}