#include "hsys/vector_ops.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

constexpr float kPrec = 1e-6F;

hsys::Vector<float> to_device(const Eigen::VectorXf& host) {
  hsys::Vector<float> v(host.size());
  v.data().copy_from_host(host.data());
  return v;
}

Eigen::VectorXf to_host(const hsys::Vector<float>& v) {
  Eigen::VectorXf host(v.size());
  v.data().copy_to_host(host.data());
  return host;
}

class VecAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecAddTest, MatchesEigen) {
  const auto n = static_cast<Eigen::Index>(GetParam());
  std::srand(static_cast<unsigned>(n));

  const Eigen::VectorXf a_host = Eigen::VectorXf::Random(n);
  const Eigen::VectorXf b_host = Eigen::VectorXf::Random(n);
  const Eigen::VectorXf expected = a_host + b_host;

  const auto a = to_device(a_host);
  const auto b = to_device(b_host);
  const auto c = a + b;

  ASSERT_EQ(c.size(), GetParam());
  EXPECT_TRUE(to_host(c).isApprox(expected, kPrec));
}

INSTANTIATE_TEST_SUITE_P(
    Sizes, VecAddTest, ::testing::Values(1, 2, 3, 127, 128, 129, 512, 1024, 1029));

TEST(VecAddEdge, SizeMismatchThrows) {
  const hsys::Vector<float> a(10);
  const hsys::Vector<float> b(11);
  EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(VecAddEdge, EmptyVectors) {
  const hsys::Vector<float> a(0);
  const hsys::Vector<float> b(0);
  EXPECT_EQ((a + b).size(), 0U);
}

TEST(DataTest, CopyIsDeep) {
  const std::vector<int> src{1, 2, 3, 4};
  hsys::Data<int> d1(src.size());
  d1.copy_from_host(src.data());

  hsys::Data<int> d2 = d1;
  EXPECT_NE(d1.data(), d2.data());

  // меняем первый буфер, второй остаться прежним
  const std::vector<int> other{9, 9, 9, 9};
  d1.copy_from_host(other.data());

  std::vector<int> out(src.size());
  d2.copy_to_host(out.data());
  EXPECT_EQ(out, src);
}

TEST(DataTest, MoveLeavesSourceEmpty) {
  hsys::Data<int> d1(8);
  const int* ptr = d1.data();

  hsys::Data<int> d2 = std::move(d1);
  EXPECT_EQ(d2.data(), ptr);
  EXPECT_EQ(d2.size(), 8U);
  EXPECT_EQ(d1.data(), nullptr);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(d1.size(), 0U);  // NOLINT(bugprone-use-after-move)

  hsys::Data<int> d3(2);
  d3 = std::move(d2);
  EXPECT_EQ(d3.data(), ptr);
}

TEST(DataTest, CopyAssignment) {
  const std::vector<int> src{5, 6, 7};
  hsys::Data<int> d1(src.size());
  d1.copy_from_host(src.data());

  hsys::Data<int> d2(1);
  d2 = d1;
  ASSERT_EQ(d2.size(), src.size());

  std::vector<int> out(src.size());
  d2.copy_to_host(out.data());
  EXPECT_EQ(out, src);
}

TEST(VectorTest, CopySharesData) {
  hsys::Vector<float> a(16);
  const hsys::Vector<float> b = a;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(&a.data(), &b.data());
}

TEST(VectorTest, ViewIsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<hsys::VectorView<float>>);
  EXPECT_TRUE(std::is_trivially_copyable_v<hsys::VectorView<double>>);
}

}  // namespace
