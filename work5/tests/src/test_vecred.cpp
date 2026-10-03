#include "hsys/vector_reduce.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>

namespace {

constexpr float kPrec = 1e-4F;

enum class Kernel { NoBroadcast, Broadcast };

std::unique_ptr<hsys::VecredStrategy<float>> make_strategy(Kernel kernel) {
  if (kernel == Kernel::NoBroadcast) return std::make_unique<hsys::VecredShared<float>>();
  return std::make_unique<hsys::VecredShuffle<float>>();
}

hsys::Vector<float> to_device(const Eigen::VectorXf& host) {
  hsys::Vector<float> v(host.size());
  v.data().copy_from_host(host.data());
  return v;
}

using Param = std::tuple<Kernel, std::size_t>;

class VecredTest : public ::testing::TestWithParam<Param> {};

std::string param_name(const ::testing::TestParamInfo<Param>& info) {
  const auto [kernel, n] = info.param;
  const std::string prefix = kernel == Kernel::NoBroadcast ? "nobr_" : "br_";
  return prefix + std::to_string(n);
}

TEST_P(VecredTest, MatchesEigen) {
  const auto [kernel, n] = GetParam();
  std::srand(static_cast<unsigned>(n));

  const Eigen::VectorXf host = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
  const auto v = to_device(host);

  EXPECT_NEAR(hsys::sum(v, *make_strategy(kernel)), host.sum(), kPrec);
}

INSTANTIATE_TEST_SUITE_P(Sizes,
    VecredTest,
    ::testing::Combine(::testing::Values(Kernel::NoBroadcast, Kernel::Broadcast),
        ::testing::Values<std::size_t>(1, 2, 3, 127, 129, 512, 541, 1037)),
    param_name);

// большой вектор: блоков больше max_blocks, работает цикл с шагом сетки
TEST_P(VecredTest, LargeVectorOfOnes) {
  const auto kernel = std::get<0>(GetParam());
  const std::size_t n = std::size_t{1} << 21;

  const Eigen::VectorXf host = Eigen::VectorXf::Ones(static_cast<Eigen::Index>(n));
  const auto v = to_device(host);

  EXPECT_EQ(hsys::sum(v, *make_strategy(kernel)), static_cast<float>(n));
}

TEST(VecredEdge, EmptyVectorGivesZero) {
  const hsys::Vector<float> v(0);
  EXPECT_EQ(hsys::sum(v, hsys::VecredShared<float>{}), 0.0F);
  EXPECT_EQ(hsys::sum(v, hsys::VecredShuffle<float>{}), 0.0F);
}

TEST(VecredEdge, SmallWorkspaceThrows) {
  const hsys::Vector<float> v(100000);
  hsys::Vector<float> workspace(1);
  hsys::Vector<float> out(1);
  EXPECT_THROW(hsys::reduce(v, workspace, out, hsys::VecredShuffle<float>{}),
      std::invalid_argument);
}

}  // namespace
