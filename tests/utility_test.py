import numpy as np
import Comp_Quant_Dynam.utility as util


class Test_example_function:

    def test_example_func_zero(self):
        x = 0
        expected =  1 / np.pi ** (1 / 4)
        result = util.example_func(x)
        assert np.allclose(expected, result)

    def test_example_func_symmetry(self):
        x = np.array([-1, 1])
        result = util.example_func(x)
        assert np.allclose(result[0], result[1])


#################### Solution sheet 2 ####################


class Test_create_xvals:

    L = 10
    npoints = 101
    
    def test_create_xvals_length(self):
        
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert len(xvals) == self.npoints

    def test_create_xvals_range(self):

        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[0], -self.L/2)
        assert np.isclose(xvals[-1], self.L/2)

    def test_create_xvals_spacing(self):
        xvals, dx = util.create_xvals(self.L, self.npoints)
        expected_dx = self.L / (self.npoints - 1)
        assert np.isclose(dx, expected_dx)

    def test_create_xvals_zero_centered(self):
        # only works if npoints is odd
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[self.npoints // 2], 0) # check that the middle point is approximately zero
