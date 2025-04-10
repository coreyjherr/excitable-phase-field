#include <catch2/catch_test_macros.hpp>
#include "model.h"

TEST_CASE("Phase Field Initialization", "[phase_field_initialization]")
{
    // Test case for phase field initialization
    // This is a placeholder test case. You can add your own test logic here.
    REQUIRE(true == true);

    // Example: Check if the phase field is initialized correctly
    Param p;
    p.alpha=0;
    p.T=10000;
    Model model = Model(p, 0, "./tmp/", 0, "", false);
    model.simulate(false, false);
    // Check if the phase field is initialized correctly
    // Real com_x = model.get_com_x();
    // Real com_y = model.get_com_y();

}
