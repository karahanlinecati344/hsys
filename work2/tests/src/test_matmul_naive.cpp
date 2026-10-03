#include "hsys/matrix_ops.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

// Eigen по умолчанию хранит матрицы по столбцам, а у нас row-major
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float kPrec = 1e-5F;

hsys::Matrix<float> to_device(const RowMatrixXf& host) {
  hsys::Matrix<float> m(host.rows(), host.cols());
  m.data().copy_from_host(host.data());
  return m;
}

RowMatrixXf to_host(const hsys::Matrix<float>& m) {
  RowMatrixXf host(m.nrows(), m.ncols());
  m.data().copy_to_host(host.data());
  return host;
}

using Dims = std::tuple<std::size_t, std::size_t, std::size_t>;

class MatmulNaiveTest : public ::testing::TestWithParam<Dims> {};

std::string dims_name(const ::testing::TestParamInfo<Dims>& info) {
  const auto [m, n, k] = info.param;
  return "m" + std::to_string(m) + "_n" + std::to_string(n) + "_k" + std::to_string(k);
}

TEST_P(MatmulNaiveTest, MatchesEigen) {
  const auto [m, n, k] = GetParam();
  std::srand(static_cast<unsigned>(m * 1000 + n * 10 + k));

  const RowMatrixXf a_host = RowMatrixXf::Random(m, k);
  const RowMatrixXf b_host = RowMatrixXf::Random(k, n);
  const Eigen::MatrixXf expected = a_host * b_host;

  const auto a = to_device(a_host);
  const auto b = to_device(b_host);
  const auto c = a * b;

  ASSERT_EQ(c.nrows(), m);
  ASSERT_EQ(c.ncols(), n);
  EXPECT_TRUE(to_host(c).isApprox(expected, kPrec));
}

const auto kSizes = ::testing::Values<std::size_t>(1, 2, 3, 127, 128, 129, 512);

INSTANTIATE_TEST_SUITE_P(
    Sizes, MatmulNaiveTest, ::testing::Combine(kSizes, kSizes, kSizes), dims_name);

TEST(MatmulNaiveEdge, SizeMismatchThrows) {
  const hsys::Matrix<float> a(3, 4);
  const hsys::Matrix<float> b(5, 2);
  EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatmulNaiveEdge, ResultKeepsStrategy) {
  const hsys::Matrix<float> a(2, 2);
  const hsys::Matrix<float> b(2, 2);
  const auto c = a * b;
  EXPECT_EQ(c.matmul(), a.matmul());
}

TEST(MatrixTest, CopySharesData) {
  hsys::Matrix<float> a(4, 8);
  const hsys::Matrix<float> b = a;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(&a.data(), &b.data());
  EXPECT_EQ(b.nrows(), 4U);
  EXPECT_EQ(b.ncols(), 8U);
  EXPECT_EQ(b.size(), 32U);
}

TEST(MatrixTest, ViewIsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<hsys::MatrixView<float>>);
  EXPECT_TRUE(std::is_trivially_copyable_v<hsys::MatrixView<double>>);
}

}  // namespace
