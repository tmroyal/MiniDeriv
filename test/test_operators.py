import unittest

import numpy as np

from operators import dudx, laplacian, d3udx3, d4udx4


class TestOperators(unittest.TestCase):
    def test_dudx(self):
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dx = 1.0
        expected_derivative = 1.0

        # iterate through each index not at the boundaries
        for i in range(1, len(u) - 1):
            result = dudx(u, i, dx)
            self.assertAlmostEqual(result, expected_derivative, places=6)

        # test the first and last index with periodic boundary conditions
        expected_boundary = -1.5
        result_first = dudx(u, 0, dx)
        result_last = dudx(u, len(u) - 1, dx)
        self.assertAlmostEqual(result_first, expected_boundary, places=6)
        self.assertAlmostEqual(result_last, expected_boundary, places=6)

    def test_laplacian(self):
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dx = 1.0
        expected_laplacian = 0.0

        # iterate through each index not at the boundaries
        for i in range(1, len(u) - 1):
            result = laplacian(u, i, dx)
            self.assertAlmostEqual(result, expected_laplacian, places=6)
        # test the first and last index with periodic boundary conditions

        result_first = laplacian(u, 0, dx)
        result_last = laplacian(u, len(u) - 1, dx)
        self.assertAlmostEqual(result_first, 5.0, places=6)
        self.assertAlmostEqual(result_last, -5.0, places=6)

        # test with parabola
        u_par = np.array([x**2 for x in range(5)])
        expected_laplacian_par = 2.0
        for i in range(1, len(u_par) - 1):
            result = laplacian(u_par, i, dx)
            self.assertAlmostEqual(result, expected_laplacian_par, places=6)

        result_first_par = laplacian(u_par, 0, dx)
        result_last_par = laplacian(u_par, len(u_par) - 1, dx)
        self.assertAlmostEqual(result_first_par, 17.0, places=6)
        self.assertAlmostEqual(result_last_par, -23.0, places=6)

    def test_d3udx3(self):
        # test with parabolic
        u = np.array([x**3 for x in range(5)])
        dx = 1.0
        expected_third_derivative = 6.0

        # iterate through each index not at the boundaries
        # with central third, that is two places
        for i in range(2, len(u) - 2):
            result = d3udx3(u, i, dx)
            self.assertAlmostEqual(result, expected_third_derivative, places=6)

        # test first and last two indices with periodic boundary conditions
        expected_boundaries = [53.5, -26.5, -56.5, 23.5]
        result_first = d3udx3(u, 0, dx)
        result_second = d3udx3(u, 1, dx)
        result_penultimate = d3udx3(u, len(u) - 2, dx)
        result_last = d3udx3(u, len(u) - 1, dx)
        self.assertAlmostEqual(result_first, expected_boundaries[0], places=6)
        self.assertAlmostEqual(result_second, expected_boundaries[1], places=6)
        self.assertAlmostEqual(result_penultimate, expected_boundaries[2], places=6)
        self.assertAlmostEqual(result_last, expected_boundaries[3], places=6)

    def test_d4udx4(self):
        # test with quartic
        u = np.array([x**4 for x in range(5)])
        dx = 1.0
        expected_fourth_derivative = 24.0

        # iterate through each index not at the boundaries
        # with central fourth, that is three places
        for i in range(3, len(u) - 3):
            result = d4udx4(u, i, dx)
            self.assertAlmostEqual(result, expected_fourth_derivative, places=6)

        expected_boundaries = [-931, 279, -601, 1229]
        result_first = d4udx4(u, 0, dx)
        result_second = d4udx4(u, 1, dx)
        result_penultimate = d4udx4(u, len(u) - 2, dx)
        result_last = d4udx4(u, len(u) - 1, dx)
        self.assertAlmostEqual(result_first, expected_boundaries[0], places=6)
        self.assertAlmostEqual(result_second, expected_boundaries[1], places=6)
        self.assertAlmostEqual(result_penultimate, expected_boundaries[2], places=6)
        self.assertAlmostEqual(result_last, expected_boundaries[3], places=6)
