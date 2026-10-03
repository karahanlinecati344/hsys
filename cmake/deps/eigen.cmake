# Если Eigen уже установлен в системе - берём его, иначе скачиваем
find_package(Eigen3 3.4 QUIET NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
  include(FetchContent)

  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)
endif()
