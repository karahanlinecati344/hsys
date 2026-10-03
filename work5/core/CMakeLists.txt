add_library(work5_core STATIC src/vecred.cu)

target_include_directories(work5_core PUBLIC include)
target_link_libraries(work5_core PUBLIC work1_core)

cuda_setup_targets(
  TARGETS work5_core
  CUDA_MIN_ARCH ${HSYS_CUDA_MIN_ARCH}
)
