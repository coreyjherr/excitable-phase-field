#include <catch2/catch_test_macros.hpp>
#include "model.h"
#include "params.hpp"

TEST_CASE("Phase Field Initialization", "[phase_field_initialization]") {
  // Test case for phase field initialization
  // This is a placeholder test case. You can add your own test logic here.
  REQUIRE(true == true);

  // Example: Check if the phase field is initialized correctly
  Param p;
  p.alpha = 0;
  p.xi = 5.0;
  p.gamma = 2.5;
  p.T = 10000;
  Real area = 3.14159 * p.r0 * p.r0 / p.dx / p.dx;

  Model model = Model(p, 0, "./tmp/", 0, "", WRITE_NONE);
  model.updateArea();
  model.updateCOM();

  REQUIRE(std::abs(model.getArea()[0] - area) / area < 5e-2);

  Real com_x = model.getCOMX()[0];
  Real com_y = model.getCOMY()[0];
  model.simulate(false);
  Real com_x_after = model.getCOMX()[0];
  Real com_y_after = model.getCOMY()[0];

  // Check that the center of mass has not changed
  REQUIRE(std::abs(com_x - com_x_after) < 1e-4);
  REQUIRE(std::abs(com_y - com_y_after) < 1e-4);

  // Check that the area is still close to the expected value
  REQUIRE(std::abs(model.getArea()[0] - area) / area < 5e-2);
}
