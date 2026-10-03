#include "hsys/matrix_ops.cuh"
#include "hsys/naive_matmul.cuh"
#include "hsys/shmem_matmul.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>

namespace {

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float kPrec = 1e-5F;

const auto kShmem = std::make_shared<const hsys::ShmemMatmul<float>>();

hsys::Matrix<float> to_device(const RowMatrixXf& host) {
  hsys::Matrix<float> m(host.rows(), host.cols(), kShmem);
  m.data().copy_from_host(host.data());
  return m;
}

RowMatrixXf to_host(const hsys::Matrix<float>& m) {
  RowMatrixXf host(m.nrows(), m.ncols());
  m.data().copy_to_host(host.data());
  return host;
}

using Dims = std::tuple<std::size_t, std::size_t, std::size_t>;

class MatmulShmemTest : public ::testing::TestWithParam<Dims> {};

std::string dims_name(const ::testing::TestParamInfo<Dims>& info) {
  const auto [m, n, k] = info.param;
  return "m" + std::to_string(m) + "_n" + std::to_string(n) + "_k" + std::to_string(k);
}

TEST_P(MatmulShmemTest, MatchesEigen) {
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
    Sizes, MatmulShmemTest, ::testing::Combine(kSizes, kSizes, kSizes), dims_name);

TEST(MatmulShmemEdge, SizeMismatchThrows) {
  const hsys::Matrix<float> a(3, 4, kShmem);
  const hsys::Matrix<float> b(5, 2, kShmem);
  EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatmulShmemEdge, SameAsNaive) {
  std::srand(42);
  const RowMatrixXf a_host = RowMatrixXf::Random(70, 45);
  const RowMatrixXf b_host = RowMatrixXf::Random(45, 33);

  auto a = to_device(a_host);
  const auto b = to_device(b_host);
  const RowMatrixXf by_shmem = to_host(a * b);

  // меняем стратегию у той же матрицы и считаем ещё раз
  a.set_matmul(std::make_shared<const hsys::NaiveMatmul<float>>());
  const RowMatrixXf by_naive = to_host(a * b);

  EXPECT_TRUE(by_shmem.isApprox(by_naive, kPrec));
}

}  // namespace
