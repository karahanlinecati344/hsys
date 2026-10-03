#include "hsys/matrix_ops.cuh"
#include "hsys/wmma_matmul.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>

namespace {

using RowMatrixXh
    = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Eigen::half и __half - оба 16-битные IEEE, поэтому можно копировать байты как есть
static_assert(sizeof(Eigen::half) == sizeof(__half));

constexpr float kPrec = 1e-2F;

const auto kWmma = std::make_shared<const hsys::WmmaMatmul>();

hsys::Matrix<__half> to_device(const RowMatrixXh& host) {
  hsys::Matrix<__half> m(host.rows(), host.cols(), kWmma);
  m.data().copy_from_host(reinterpret_cast<const __half*>(host.data()));
  return m;
}

RowMatrixXh to_host(const hsys::Matrix<__half>& m) {
  RowMatrixXh host(m.nrows(), m.ncols());
  m.data().copy_to_host(reinterpret_cast<__half*>(host.data()));
  return host;
}

RowMatrixXh random_half(Eigen::Index rows, Eigen::Index cols) {
  return Eigen::MatrixXf::Random(rows, cols).cast<Eigen::half>();
}

using Dims = std::tuple<std::size_t, std::size_t, std::size_t>;

class MatmulWmmaTest : public ::testing::TestWithParam<Dims> {};

std::string dims_name(const ::testing::TestParamInfo<Dims>& info) {
  const auto [m, n, k] = info.param;
  return "m" + std::to_string(m) + "_n" + std::to_string(n) + "_k" + std::to_string(k);
}

TEST_P(MatmulWmmaTest, MatchesEigen) {
  const auto [m, n, k] = GetParam();
  std::srand(static_cast<unsigned>(m * 1000 + n * 10 + k));

  const RowMatrixXh a_host = random_half(m, k);
  const RowMatrixXh b_host = random_half(k, n);
  const RowMatrixXh expected = a_host * b_host;

  const auto a = to_device(a_host);
  const auto b = to_device(b_host);
  const auto c = a * b;

  ASSERT_EQ(c.nrows(), m);
  ASSERT_EQ(c.ncols(), n);

  const Eigen::MatrixXf result = to_host(c).cast<float>();
  EXPECT_TRUE(result.isApprox(expected.cast<float>(), kPrec));
}

const auto kSizes = ::testing::Values<std::size_t>(16, 32, 64, 128, 256, 512);

INSTANTIATE_TEST_SUITE_P(
    Sizes, MatmulWmmaTest, ::testing::Combine(kSizes, kSizes, kSizes), dims_name);

// размеры не кратны 16 - работают за счёт дополнения тайлов нулями
INSTANTIATE_TEST_SUITE_P(Unaligned,
    MatmulWmmaTest,
    ::testing::Values(Dims{1, 1, 1}, Dims{17, 33, 5}, Dims{100, 70, 129}),
    dims_name);

TEST(MatmulWmmaEdge, SizeMismatchThrows) {
  const hsys::Matrix<__half> a(16, 32, kWmma);
  const hsys::Matrix<__half> b(16, 16, kWmma);
  EXPECT_THROW(a * b, std::invalid_argument);
}

}  // namespace
