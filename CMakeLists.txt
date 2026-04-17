cmake_minimum_required(VERSION 3.10)

# Save file with path to simulation sources
set(SIM_SRC_DIR "sim_src_dir.txt")
execute_process(
  COMMAND echo ${CMAKE_CURRENT_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SIM_SRC_DIR}"
)

# Create directories for simulation outputs and figures
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/simulation_outputs")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/figures_and_plots")

# Executables (add_simulation is provided by top-level UGUCA CMakeLists)
add_simulation(time_int_opt time_int_opt.cc)
add_simulation(dd_earthquake dd_earthquake.cc)

# Optional target if present
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rs_sw_regular_dumping.cc")
  add_simulation(rs_sw_regular_dumping rs_sw_regular_dumping.cc)
endif()

# Copy full simulation machinery needed at runtime (no loops)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plotting_tools")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/plotting_tools"
       DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/input_files")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/input_files"
       DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/analysis_and_processing_tools")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/analysis_and_processing_tools"
       DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/automation")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/automation"
       DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Root-level runtime helpers
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Plot_BaselineExp_Spacetime.py")
  configure_file(Plot_BaselineExp_Spacetime.py Plot_BaselineExp_Spacetime.py COPYONLY)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/compute_baseline_vs_exp_rmse.py")
  configure_file(compute_baseline_vs_exp_rmse.py compute_baseline_vs_exp_rmse.py COPYONLY)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  configure_file(README.md README.md COPYONLY)
endif()
