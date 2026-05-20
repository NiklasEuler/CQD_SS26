import numpy as np
import numpy.linalg as LA
import pytest
import scipy.sparse as sparse

import Comp_Quant_Dynam.hamiltonians as ham
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.unitaries as unitaries
import Comp_Quant_Dynam.operators as ops
import Comp_Quant_Dynam.integrators as integrators


##################### Exercise sheet 6 ####################


class Test_integrate_ODE:
    
    def test_integrate_ODE_ED(self):
        N = 4
        state = np.eye(1, 2**N, 6).flatten() # initial state |0110>
        sig_x = ops.sigma_x_sparse() # single-site sigma_x operator
        sig_y = ops.sigma_y_sparse() # single-site sigma_y operator
        
        local_dims = [2] * N
        H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x) + 0.5 * ops.n_party_op_sparse(local_dims, 2, sig_y)
        
        t_steps = 100
        t_end = 2 * np.pi
        tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)

        obsv_vec = [ops.n_party_op_sparse(local_dims, 3, ops.sigma_z_sparse())] # observable: sigma_z on site 3

        observables_ED = unitaries.calc_expv_ED(obsv_vec, H, state, tvec)
        stepper_func = integrators.Euler_step
        int_steps_per_dtout = 500
        stepper_args = None
        observables_int = integrators.integrate_ODE(stepper_func, obsv_vec, H, state, tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(observables_ED - observables_int)))
        assert np.allclose(observables_ED, observables_int, atol = 1e-3)

    def test_integrate_ODE_warnings(self):
        N = 4
        state = np.eye(1, 2**N, 8).flatten() # initial state |1000>
        sig_x = ops.sigma_x_sparse() # single-site sigma_x operator
        sig_z = ops.sigma_z_sparse() # single-site sigma_z operator

        local_dims = [2] * N
        non_hermitian_op = sig_x + 1j * sig_z # non-Hermitian operator
        nh3 = ops.n_party_op_sparse(local_dims, 3, non_hermitian_op) # non-Hermitian operator acting on site 3
        
        H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x)
        
        t_steps = 100
        t_end = 2 * np.pi
        tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)
    
        stepper_func = integrators.Euler_step
        obsv_vec = [nh3] # non-Hermitian observable acting on site 3
        pytest.warns(UserWarning, integrators.integrate_ODE, stepper_func, obsv_vec, H, state, tvec, int_steps_per_dtout=10, stepper_args=None)

        