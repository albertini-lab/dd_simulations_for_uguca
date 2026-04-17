/**
 * @file   time_int_opt.cc
 *
 * @author David S. Kammer <dkammer@ethz.ch>
 * @author Gabriele Albertini <ga288@cornell.edu>
 * @author Chun-Yu Ke <ck659@cornell.edu>
 * @author Joshua McNeely <pmyjm22@nottingham.ac.uk>
 * @date creation: Thur 3 Apr 2025
 * @date last modification: Wed 21 May 2025
 *
 * @brief  Simulation demonstrating time integration options including
 *         ground truth generation and data-driven methods.
 *         Simulates a bimaterial interface under shear and normal load.
 *         Modes:
 *         1. Ground Truth Generation: Standard dynamic simulation saving
 *            interface displacements at every time step.
 *         2. Data-Driven Simulation: Uses pre-computed ground truth
 *            displacements within computeDisplacementWithData method.
 *
 * Copyright (C) 2021 ETH Zurich (David S. Kammer)
 *
 * This file is part of uguca.
 *
 * uguca is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * uguca is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with uguca.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "static_communicator_mpi.hh"
#include "uca_parameter_reader.hh"
#include "material.hh"
#include "uca_simple_mesh.hh"
#include "half_space.hh"
#include "nodal_field.hh"
#include "uca_restart.hh"
#include "bimat_interface.hh"
#include "barras_law.hh"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <optional>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <chrono>
#include <iomanip>  // Add this for formatting

namespace fs = std::filesystem;

using namespace uguca;

// Helper function to create organized output directory structure
std::string createOutputDirectory(const std::string& sim_name, const std::string& sim_type, int world_rank) {
  std::string timestamp = ""; // Could add timestamp if needed for uniqueness
  std::string output_base = "simulation_outputs";
  std::string full_output_dir = output_base + "/" + sim_name + "_" + sim_type;
  
  if (world_rank == 0) {
    try {
      if (!fs::exists(output_base)) {
        fs::create_directory(output_base);
      }
      if (!fs::exists(full_output_dir)) {
        fs::create_directories(full_output_dir);
        std::cout << "Created output directory: " << full_output_dir << std::endl;
      } else {
        std::cout << "Output directory already exists: " << full_output_dir << std::endl;
      }
    } catch (const fs::filesystem_error& e) {
      std::cerr << "Error creating output directory " << full_output_dir << ": " << e.what() << std::endl;
      throw;
    }
  }
  
  return full_output_dir;
}

int main(int argc, char *argv[]) {

  // MPI Setup
  StaticCommunicatorMPI * comm = StaticCommunicatorMPI::getInstance();
  int world_rank = comm->whoAmI();

  // Input File Handling - Now looks in input_files directory
  std::string fname;
  if(argc<2) {
    if (world_rank==0) {
      std::cerr << "Usage: ./time_int_opt <input_file>" << std::endl;
      std::cerr << "  Example Ground Truth: ./time_int_opt ground_truth.in" << std::endl;
      std::cerr << "  Example Data-Driven:  ./time_int_opt data_driven.in" << std::endl;
      std::cerr << "  Example Noisy Data-Driven: ./time_int_opt data_driven_noisy.in" << std::endl;
      std::cerr << "  Note: Input files are expected to be in the 'input_files/' directory" << std::endl;
    }
    comm->barrier(); 
    return 1;
  }
  else {
    std::string input_file = argv[1];
    // Check if the file already includes the path
    if (input_file.find("input_files/") == 0) {
      fname = input_file;
    } else {
      // Add input_files/ prefix if not present
      fname = "input_files/" + input_file;
    }
    
    // Verify the file exists
    if (!fs::exists(fname)) {
      if (world_rank == 0) {
        std::cerr << "ERROR: Input file not found: " << fname << std::endl;
        std::cerr << "Please ensure the input file exists in the 'input_files/' directory" << std::endl;
      }
      comm->barrier();
      return 1;
    }
  }

  ParameterReader data;
  data.readInputFile(fname);

  bool ground_truth_mode = data.getOrUse<bool>("ground_truth_solution_generation", false);
  bool data_driven_active = data.getOrUse<bool>("enable_data_driven_mode", false);

  // Mesh Setup
  double length   = data.get<double>("length");
  int nb_elements = data.getOrUse<int>("nb_elements", 1024);
  SimpleMesh mesh(length, nb_elements);
  const int P_local_nodes = mesh.getNbLocalNodes(); 

  // Constitutive Law
  BarrasLaw law(mesh,
        data.get<double>("tauc", "cohesion"),
        data.get<double>("dc", "cohesion"));

  // Material Properties
  Material top_mat = Material(data.get<double>("E","top"),
                  data.get<double>("nu","top"),
                  data.get<double>("rho","top"));
  top_mat.readPrecomputedKernels();
  Material bot_mat = Material(data.get<double>("E","bot"),
                  data.get<double>("nu","bot"),
                  data.get<double>("rho","bot"));
  bot_mat.readPrecomputedKernels();

  // Interface Setup
  BimatInterface interface(mesh, {_x,_y}, top_mat, bot_mat, law);

  interface.getLoad().setAllValuesTo(data.get<double>("shear_load"), _x);
  interface.getLoad().setAllValuesTo(data.get<double>("normal_load"), _y);

  // Time Stepping Parameters
  double duration = data.get<double>("duration");
  double time_step = data.get<double>("tsf") * interface.getStableTimeStep();
  int step_advances = data.getOrUse<int>("step_advances", 1);
  int nb_time_steps = static_cast<int>(duration / (time_step * step_advances));
  interface.setTimeStep(time_step);

  interface.init();

  int data_driven_update_interval = 1;

  // Fields for Data-Driven Mode
  NodalField top_u_data(mesh,{_x,_y},"top_u_data");
  NodalField bot_u_data(mesh,{_x,_y},"bot_u_data");
  NodalField top_w_factor(mesh,{_x,_y},"top_w_factor"); 
  NodalField bot_w_factor(mesh,{_x,_y},"bot_w_factor"); 
  NodalField loaded_top_weights(mesh,{_x,_y},"loaded_top_weights"); 
  NodalField loaded_bot_weights(mesh,{_x,_y},"loaded_bot_weights"); 
  std::optional<Restart> ground_truth_loader;
  double w_factor_value = 0.0;
  double noise_variance = 0.0;
  std::optional<Restart> calculated_disp_dumper;
  std::string calculated_disp_dump_output_dir_path;

  // Fields for Standard Simulation Per-Step Displacement Dump
  std::optional<Restart> standard_sim_disp_dumper;
  std::string standard_sim_disp_dump_output_dir_path;

  // Setup specific to Data-Driven Mode
  if (data_driven_active) {
      if (ground_truth_mode) {
          if (world_rank == 0) {
              std::cerr << "WARNING: 'enable_data_driven_mode' is true but 'ground_truth_solution_generation' is also true." << std::endl;
              std::cerr << "         Data-driven mode will NOT be activated when generating ground truth." << std::endl;
          }
          data_driven_active = false;
      } else {
          data_driven_update_interval = std::max(1, data.getOrUse<int>("data_driven_update_interval", 1));
          if (world_rank == 0) {
              std::cout << "-----------------------------------------" << std::endl;
              std::cout << "--- Data-Driven Simulation Mode Setup ---" << std::endl;
              std::cout << "Data-driven specific update will occur every " << data_driven_update_interval << " step(s)." << std::endl;
              if (data_driven_update_interval > 1) {
                  std::cout << "On other steps, a standard dynamic update will be performed." << std::endl;
              }
          }
          w_factor_value = data.get<double>("w_factor_constant");
          double noise_stddev_sim = data.get<double>("noise_stddev_sim"); 
          noise_variance = noise_stddev_sim * noise_stddev_sim;          

          // Update the ground truth data path to look in simulation_outputs if it doesn't start with a known prefix
          std::string gt_data_path = data.get<std::string>("ground_truth_data_path");
          if (gt_data_path.find("simulation_outputs/") != 0 && gt_data_path.find("./") != 0 && gt_data_path.find("/") != 0) {
            gt_data_path = "simulation_outputs/" + gt_data_path;
          }
          std::string gt_data_name = data.get<std::string>("ground_truth_data_name");

          if (!fs::exists(gt_data_path)) {
               if (world_rank == 0) {
                   std::cerr << "ERROR: Ground truth data path not found: " << gt_data_path << std::endl;
                   std::cerr << "Ensure 'ground_truth_data_path' points to the directory created by the ground truth run." << std::endl;
               }
               comm->barrier();
               return 1;
          }

          // Change from ASCII to Binary format for smaller files
          ground_truth_loader.emplace(gt_data_name, gt_data_path, Restart::Format::Binary);
          ground_truth_loader->registerIO("interface_top_disp", top_u_data);
          ground_truth_loader->registerIO("interface_bot_disp", bot_u_data);
          ground_truth_loader->registerIO("interface_top_weights", loaded_top_weights);
          ground_truth_loader->registerIO("interface_bot_weights", loaded_bot_weights);

          // Enable calculated displacement dumping for data-driven mode for analysis
          calculated_disp_dump_output_dir_path = createOutputDirectory(data.get<std::string>("sim_name"), "data_driven", world_rank);
          std::string cdd_sim_name = data.get<std::string>("sim_name") + "_displacement_dump";
          
          comm->barrier(); 

          calculated_disp_dumper.emplace(cdd_sim_name, calculated_disp_dump_output_dir_path, Restart::Format::Binary);
          calculated_disp_dumper->registerIO("interface_top_disp", interface.getTop().getDisp());
          calculated_disp_dumper->registerIO("interface_bot_disp", interface.getBot().getDisp());

          if (world_rank == 0) {
              std::cout << "Data-driven mode ACTIVE." << std::endl;
              std::cout << "Attempting to load ground truth fields 'interface_top_disp', 'interface_bot_disp'," << std::endl;
              std::cout << "  and weight fields 'interface_top_weights', 'interface_bot_weights'." << std::endl;
              std::cout << "from path: " << gt_data_path << " using base name: " << gt_data_name << std::endl;
              std::cout << "(Restart mechanism expects data in: " << gt_data_path << "/" << gt_data_name << "-restart/)" << std::endl;
              std::cout << "Using global w_factor_constant = " << w_factor_value << " (to be multiplied by loaded weights)." << std::endl;
              std::cout << "Using noise_stddev_sim = " << noise_stddev_sim << " -> noise_variance (sigma_sq) = " << noise_variance << " for formula." << std::endl;
              std::cout << "P_local_nodes (local nodes on this rank) = " << P_local_nodes << std::endl;

              double N_global_val = static_cast<double>(mesh.getNbGlobalNodes()); 
              double P_term_for_illustration = noise_variance; 
              double dt_eff_calc = time_step * step_advances;
              double w_i_calc_for_print = w_factor_value; 

              double w_hat_calc = N_global_val * dt_eff_calc * w_i_calc_for_print;
              double denominator_calc = w_hat_calc * dt_eff_calc + P_term_for_illustration;

              std::cout << "Calculated N_global = " << N_global_val << std::endl;
              std::cout << "Calculated dt_eff (time_step * step_advances) = " << dt_eff_calc << std::endl;
              std::cout << "Calculated w_hat (N_global * dt_eff * w_factor_constant) = " << w_hat_calc << std::endl;

              if (std::abs(denominator_calc) < 1e-15) {
                  std::cout << "Denominator for factors is near zero, factors are ill-defined." << std::endl;
                  std::cout << "Calculated explicit_disp_factor = N/A (denominator zero)" << std::endl;
                  std::cout << "Calculated u_data_factor = N/A (denominator zero)" << std::endl;
              } else {
                  double explicit_disp_factor_calc = P_term_for_illustration / denominator_calc; 
                  double u_data_factor_calc = (w_hat_calc * dt_eff_calc) / denominator_calc;
                  std::cout << "Illustrative explicit_disp_factor (sigma_sq / (w_hat*dt_eff + sigma_sq)) = " << explicit_disp_factor_calc << std::endl; 
                  std::cout << "Illustrative u_data_factor ((w_hat*dt_eff) / (w_hat*dt_eff + sigma_sq)) = " << u_data_factor_calc << std::endl; 
              }
              std::cout << "Calculated displacements (from chosen update method) will be dumped at every step into: " << calculated_disp_dump_output_dir_path << std::endl;
              std::cout << "-----------------------------------------" << std::endl;
          }
      }
  }

  // Setup for Standard Simulation Per-Step Displacement Dump
  bool enable_displacement_dump_standard = data.getOrUse<bool>("enable_displacement_dump_standard", false);
  if (!ground_truth_mode && !data_driven_active) {
      if (enable_displacement_dump_standard) {
          if (world_rank == 0) {
              std::cout << "------------------------------------------------------------------" << std::endl;
              std::cout << "--- Standard Simulation: Per-step displacement dump ENABLED ---" << std::endl;
              std::cout << "Displacement dumps will be generated at every time step." << std::endl;
              std::cout << "------------------------------------------------------------------" << std::endl;
          }
          
          standard_sim_disp_dump_output_dir_path = createOutputDirectory(data.get<std::string>("sim_name"), "standard", world_rank);
          std::string ssdd_sim_name = data.get<std::string>("sim_name") + "_displacement_dump";
          
          comm->barrier();

          standard_sim_disp_dumper.emplace(ssdd_sim_name, standard_sim_disp_dump_output_dir_path, Restart::Format::Binary);
          standard_sim_disp_dumper->registerIO("interface_top_disp", interface.getTop().getDisp());
          standard_sim_disp_dumper->registerIO("interface_bot_disp", interface.getBot().getDisp());
          
          if (world_rank == 0) {
              std::cout << "Displacement data will be dumped at every step into directory: " << standard_sim_disp_dump_output_dir_path << std::endl;
              std::cout << "(Restart mechanism will create files in: " << standard_sim_disp_dump_output_dir_path << "/" << ssdd_sim_name << "-restart/)" << std::endl;
          }
      } else {
          if (world_rank == 0) {
              std::cout << "------------------------------------------------------------------" << std::endl;
              std::cout << "--- Standard Simulation: Displacement dump disabled ---" << std::endl;
              std::cout << "To enable per-step displacement dumps, set 'enable_displacement_dump_standard = true'" << std::endl;
              std::cout << "------------------------------------------------------------------" << std::endl;
          }
      }
  }

   // Nucleation Site Setup
  const TwoDVector & coords = mesh.getLocalCoords();
  NodalField & tau_max = law.getTauMax();
  double a0 = data.get<double>("a0"); // Used for initial notch and potentially unilateral growth

  // --- START DYNAMIC NUCLEATION PARAMETERS ---
  bool enable_dynamic_nucleation = data.getOrUse<bool>("enable_dynamic_nucleation", false);
  double* tau_max_0nuc = nullptr;
  double dynamic_nuc_r = 0.0;
  double dynamic_nuc_tstart = 0.0;
  double dynamic_nuc_tend = 0.0;
  double dynamic_nuc_x_center = 0.0;
  double dynamic_nuc_tauc = 0.0;
  bool   dynamic_nuc_sym = true;
  double dynamic_nuc_xc0 = 0.0;
  bool   dynamic_nuc_unilateral_growth = false;

  if (enable_dynamic_nucleation) {
    if (world_rank == 0) {
        std::cout << "Dynamic nucleation enabled." << std::endl;
    }
    dynamic_nuc_r        = data.get<double>("dynamic_nuc_r");
    dynamic_nuc_tstart   = data.get<double>("dynamic_nuc_tstart");
    dynamic_nuc_tend     = data.get<double>("dynamic_nuc_tend");
    dynamic_nuc_x_center = data.get<double>("dynamic_nuc_x_center");
    dynamic_nuc_tauc     = data.get<double>("dynamic_nuc_tauc");
    dynamic_nuc_sym      = data.getOrUse<bool>("dynamic_nuc_sym", true);
    dynamic_nuc_xc0      = data.get<double>("dynamic_nuc_xc0");
    dynamic_nuc_unilateral_growth = data.getOrUse<bool>("dynamic_nuc_unilateral_growth", false);

    tau_max_0nuc = new double[mesh.getNbLocalNodes()];
    // Initialize with current tau_max values, which might be set by initial a0 notch
    for (int i = 0; i < mesh.getNbLocalNodes(); ++i) {
        tau_max_0nuc[i] = tau_max(i);
    }
  }
  // --- END DYNAMIC NUCLEATION PARAMETERS ---

  // Initial static notch (applies to both modes, a0 acts as notch_length)
  for (int i=0; i<mesh.getNbLocalNodes(); ++i) {
    if (std::abs(coords(i,0) - length/2.0) < a0/2.0) {
      tau_max(i) = 0.0;
    }
  }

  // Standard Output/Dumping Setup - Create organized output directory
  std::string sim_type_for_output = ground_truth_mode ? "ground_truth" : 
                                   (data_driven_active ? "data_driven" : "standard");
  std::string main_output_dir = createOutputDirectory(data.get<std::string>("sim_name"), sim_type_for_output, world_rank);
  
  interface.initDump(data.get<std::string>("sim_name"), main_output_dir);
  interface.registerDumpFields(data.get<std::string>("dump_fields"));
  unsigned int dump_int = std::max(1, nb_time_steps / data.getOrUse<int>("nb_dumps", 100));

  // Standard Restart (Checkpointing) Setup - DISABLED to save space
  // unsigned int restart_int = std::max(1, nb_time_steps / data.getOrUse<int>("nb_restarts", nb_time_steps));
  // Restart restart(data.get<std::string>("sim_name"), main_output_dir, Restart::Format::Binary);
  // interface.registerToRestart(restart);  // This registers ALL interface fields

  // Ground Truth Restart Setup (for GENERATION mode)
  std::optional<Restart> ground_truth_restart;
  std::string gt_output_dir_path;

  if (ground_truth_mode) {
      if (world_rank == 0) {
          std::cout << "--------------------------------------------" << std::endl;
          std::cout << "--- Ground Truth Generation Mode Setup ---" << std::endl;
      }
  
      // Create organized output directory for ground truth generation
      gt_output_dir_path = createOutputDirectory(data.get<std::string>("sim_name"), "ground_truth", world_rank);
      std::string gt_sim_name = data.get<std::string>("sim_name") + "_displacement_dump";
      
      comm->barrier();

      // Change from ASCII to Binary format for smaller files
      ground_truth_restart.emplace(gt_sim_name, gt_output_dir_path, Restart::Format::Binary);
      ground_truth_restart->registerIO(interface.getTop().getDisp()); 
      ground_truth_restart->registerIO(interface.getBot().getDisp());

      if (world_rank == 0) {
          std::cout << "Ground truth solution generation mode ACTIVE." << std::endl;
          std::cout << "Displacement data will be dumped at every step into directory: " << gt_output_dir_path << std::endl;
          std::cout << "(Restart mechanism will create files in: " << gt_output_dir_path << "/" << gt_sim_name << "-restart/)" << std::endl;
          std::cout << "--------------------------------------------" << std::endl;
      }
  }

  // Time Stepping Initialization
  int s = 1;
  double current_sim_time = 0.0;

  // Add wall clock timer
  auto sim_start_time = std::chrono::high_resolution_clock::now();

  if (data.has("restart_step")) {
    if (world_rank == 0) {
      std::cout << "Restart functionality has been disabled to save disk space." << std::endl;
      std::cout << "Please run simulation from the beginning." << std::endl;
    }
    comm->barrier();
    return 1;
  }
  else {
    s = 0;
    current_sim_time = 0.0;

    // Initial standard dump for all modes
    interface.dump(s, current_sim_time);

    if (ground_truth_mode && ground_truth_restart) {
        ground_truth_restart->dump(s);
    }
    if (!ground_truth_mode && !data_driven_active && standard_sim_disp_dumper) {
        standard_sim_disp_dumper->dump(s);
    }
    if (data_driven_active && calculated_disp_dumper) {
        calculated_disp_dumper->dump(s);
    }

    if (!ground_truth_mode) {
        if (ground_truth_loader) {
            try {
                ground_truth_loader->load(s);
                if (world_rank == 0) {
                    std::cout << "Successfully loaded initial ground truth displacement and weight data (step 0)." << std::endl;
                }
            } catch (const std::runtime_error& e) {
                if (world_rank == 0) {
                    std::cerr << "ERROR loading initial ground truth data (step 0): " << e.what() << std::endl;
                    std::cerr << "Ensure ground truth data (displacements and weights) exists for step 0 at the specified path." << std::endl;
                }
                comm->barrier();
                return 1;
            }
        }
    }
    s = 1;
  }

  // Main Time Stepping Loop
  if (world_rank == 0) {
      std::cout << "Starting time stepping loop for " << nb_time_steps << " steps..." << std::endl;
      if (ground_truth_mode) std::cout << "Mode: Ground Truth Generation" << std::endl;
      else if (data_driven_active) std::cout << "Mode: Data-Driven" << std::endl;
      else std::cout << "Mode: Standard Simulation (No GT generation, No DD)" << std::endl;
  }

  // Static mid-simulation nucleation parameters
  bool enable_mid_sim_nucleation = data.getOrUse<bool>("enable_mid_sim_nucleation", false);
  double mid_sim_nucleation_time = 0.0;
  double mid_sim_nucleation_radius = 0.0;
  bool mid_sim_nucleation_applied = false; // Flag to ensure we only apply it once
  
  if (enable_mid_sim_nucleation) {
    // Default to midpoint if not specified
    mid_sim_nucleation_time = data.getOrUse<double>("mid_sim_nucleation_time", duration / 2.0);
    mid_sim_nucleation_radius = data.getOrUse<double>("mid_sim_nucleation_radius", a0);
    
    if (world_rank == 0) {
      std::cout << "Mid-simulation static nucleation enabled." << std::endl;
      std::cout << "  Will trigger at t = " << mid_sim_nucleation_time << " s" << std::endl;
      std::cout << "  With radius = " << mid_sim_nucleation_radius << " m" << std::endl;
    }
  }

  for (; s <= nb_time_steps; ++s) {
    current_sim_time = s * time_step * step_advances;

    if (world_rank==0) {
      // Calculate elapsed wall clock time
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - sim_start_time);
      
      // Format elapsed time as HH:MM:SS
      int hours = elapsed.count() / 3600;
      int minutes = (elapsed.count() % 3600) / 60;
      int seconds = elapsed.count() % 60;
      
      std::cout << "\x1b[2K" << "Step: " << s << "/" << nb_time_steps 
                << " SimTime: " << current_sim_time 
                << " Elapsed: " << std::setfill('0') << std::setw(2) << hours << ":" 
                << std::setw(2) << minutes << ":" << std::setw(2) << seconds << "\r";
      std::cout.flush();
    }

    // --- START MID-SIMULATION NUCLEATION LOGIC ---
    if (enable_mid_sim_nucleation && !mid_sim_nucleation_applied && 
        current_sim_time >= mid_sim_nucleation_time) {
      if (world_rank == 0) {
        std::cout << std::endl << "Applying mid-simulation nucleation at t = " 
                  << current_sim_time << " s (step " << s << ")" << std::endl;
      }
      
      for (int i = 0; i < mesh.getNbLocalNodes(); ++i) {
        if (std::abs(coords(i,0) - length/2.0) < mid_sim_nucleation_radius/2.0) {
          tau_max(i) = 0.0;
        }
      }
      
      mid_sim_nucleation_applied = true;
    }
    // --- END MID-SIMULATION NUCLEATION LOGIC ---

    // --- START DYNAMIC NUCLEATION LOGIC ---
    if (enable_dynamic_nucleation) {
      // double effective_s_for_nucleation_timing = current_sim_time / time_step; // Match crack_face_forces s*time_step logic

      if (current_sim_time >= dynamic_nuc_tstart && current_sim_time <= dynamic_nuc_tend) {
        if (std::abs(current_sim_time - dynamic_nuc_tstart) < (time_step * step_advances / 2.0) || 
            (current_sim_time > dynamic_nuc_tstart && (current_sim_time - time_step * step_advances) < dynamic_nuc_tstart) ) { // Check if it's the first step at or after tstart
          // copy tau_max at the beginning of nucleation
          for (int i=0;i<mesh.getNbLocalNodes(); ++i) {
            tau_max_0nuc[i] = tau_max(i);
          }
          // unilateral propagation logic
          if (dynamic_nuc_unilateral_growth){
            for (int i=0;i<mesh.getNbLocalNodes(); ++i) {
              // Using a0 as the reference notch_length for unilateral growth
              if (coords(i,0) < length/2.0 - a0/2.0) { 
                tau_max(i) = 100.0 * tau_max_0nuc[i]; // or some large factor
              }
            }
          }
        }
        
        for (int i=0;i<mesh.getNbLocalNodes(); ++i) {
          if (!dynamic_nuc_sym) {
            if (coords(i,0) < length/2.) { // Assuming symmetry around mesh center
              continue;
            }
          }

          double px = std::abs(coords(i,0) - length/2.) - dynamic_nuc_x_center;
          double pz = 0.0; // 2D simulation
          double r_outher = std::min(std::max(0.0,
                        (current_sim_time - dynamic_nuc_tstart)/(dynamic_nuc_tend-dynamic_nuc_tstart)),
                       1.0)*dynamic_nuc_r;
          double r_inter = std::max(0.0,r_outher - dynamic_nuc_xc0);
          double radius = std::sqrt(px*px+pz*pz); // pz is 0

          if (radius < r_outher) {
            double nuc_tauc_here = dynamic_nuc_tauc;
            if (radius > r_inter && dynamic_nuc_xc0 > 1e-9) { // Avoid division by zero if xc0 is tiny
              double d = (r_outher-radius)/dynamic_nuc_xc0;
              nuc_tauc_here = std::max(dynamic_nuc_tauc, tau_max_0nuc[i] - d*(tau_max_0nuc[i]-dynamic_nuc_tauc));
            }
            tau_max(i) = std::min(tau_max(i),nuc_tauc_here);
          }
        }
      }
    }
    // --- END DYNAMIC NUCLEATION LOGIC ---

    if (ground_truth_mode) {
        interface.advanceTimeStep(_dynamic, step_advances);

        if (ground_truth_restart) {
            ground_truth_restart->dump(s); 
        }
        if (s > 0 && s % dump_int == 0) { 
            interface.dump(s, current_sim_time);
        }

    } else if (data_driven_active) {
        int ts_factor = step_advances;
        SolverMethod solver_method = _dynamic;

        if (s > 0 && s % data_driven_update_interval == 0) {
            // Perform data-driven specific update
            /*if (world_rank == 0 && (s == data_driven_update_interval || data_driven_update_interval == 1 || (s % (std::max(1, nb_time_steps/100)*data_driven_update_interval) == 0) ) ) {
                if (s == data_driven_update_interval && data_driven_update_interval > 1) std::cout << std::endl;
                std::cout << "Step " << s << ": Performing data-driven specific update." << std::endl;
            }*/

            if (ground_truth_loader) {
                try {
                    ground_truth_loader->load(s); 
                } catch (const std::runtime_error& e) {
                    if (world_rank == 0) {
                        std::cerr << std::endl << "ERROR loading ground truth data for step " << s << ": " << e.what() << std::endl;
                        std::cerr << "Check if data files (e.g., *." << s << ".* for displacements and weights) exist." << std::endl;
                    }
                    comm->barrier();
                    return 1;
                }
            } else {
                if (world_rank == 0) {
                    std::cerr << std::endl << "ERROR: Data-driven mode active but ground truth loader is not initialized!" << std::endl;
                }
                comm->barrier();
                return 1;
            }

            for (const auto& comp : loaded_top_weights.getComponents()) {
                double* loaded_weights_ptr = loaded_top_weights.data(comp);
                double* w_factor_ptr = top_w_factor.data(comp);
                for (int i = 0; i < loaded_top_weights.getNbNodes(); ++i) {
                    w_factor_ptr[i] = loaded_weights_ptr[i] * w_factor_value;
                }
            }

            for (const auto& comp : loaded_bot_weights.getComponents()) {
                double* loaded_weights_ptr = loaded_bot_weights.data(comp);
                double* w_factor_ptr = bot_w_factor.data(comp);
                for (int i = 0; i < loaded_bot_weights.getNbNodes(); ++i) {
                    w_factor_ptr[i] = loaded_weights_ptr[i] * w_factor_value;
                }
            }
        
            interface.getTop().computeDisplacementWithData(top_u_data, top_w_factor, noise_variance, false, ts_factor);
            interface.getBot().computeDisplacementWithData(bot_u_data, bot_w_factor, noise_variance, false, ts_factor);

            interface.computeInternal(false, false, solver_method, ts_factor);
            interface.computeCohesion(false, ts_factor);
            interface.computeResidual();
            interface.computeVelocity();

        } else {
            // Perform standard dynamic update
            /* if (world_rank == 0 && data_driven_update_interval > 1 && (s == 1 || (s % (std::max(1, nb_time_steps/100)*data_driven_update_interval) == 0) ) ) {
                if (s == 1) std::cout << std::endl;
                std::cout << "Step " << s << ": Performing standard dynamic update (data-driven mode)." << std::endl;
            }*/
            interface.advanceTimeStep(_dynamic, step_advances);
        }

        // Enable displacement dump for data-driven mode for analysis
        if (calculated_disp_dumper) {
            calculated_disp_dumper->dump(s);
        }

        if (s > 0 && s % dump_int == 0) { 
            interface.dump(s, current_sim_time);
        }
    } else {
        interface.advanceTimeStep(_dynamic, step_advances);

        if (standard_sim_disp_dumper) {
            standard_sim_disp_dumper->dump(s);
        }

        if (s > 0 && s % dump_int == 0) {
            interface.dump(s, current_sim_time);
        }
    }

    // Standard restart (checkpointing) for all modes - DISABLED
    // if (s > 0 && s % restart_int == 0) { 
    //   restart.dump(s);
    // }
  }

  if (world_rank == 0) {
      std::cout << std::endl << "Simulation finished successfully." << std::endl;
  }

  // Final checkpoint dump if not already done - DISABLED
  // if (nb_time_steps > 0 && nb_time_steps % restart_int != 0) {
  //     restart.dump(nb_time_steps);
  //     if (world_rank == 0) {
  //          std::cout << "Dumped final checkpoint state (step " << nb_time_steps << ")." << std::endl;
  //     }
  // }
  // Final standard dump if not covered by loop
  if (nb_time_steps > 0 && nb_time_steps % dump_int != 0) {
      interface.dump(nb_time_steps, nb_time_steps * time_step * step_advances);
       if (world_rank == 0) {
           std::cout << "Dumped final standard output state (step " << nb_time_steps << ")." << std::endl;
      }
  }

  // --- START DYNAMIC NUCLEATION CLEANUP ---
  if (tau_max_0nuc != nullptr) {
    delete[] tau_max_0nuc;
    tau_max_0nuc = nullptr;
  }
  // --- END DYNAMIC NUCLEATION CLEANUP ---

  comm->finalize();
  return 0;
}