import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat
from cardillo.solver import Solution
from cardillo.math import approx_fprime, sign

# TODO: Understand predictor of Feng mentioned in Neto1999.

# TODO: automatic increment cutting: Crisfield1991 section 9.5.1
# TODO: read Crisfield1996 section 21 Branch switching and further advanced solution procedures
# TODO: implement line searcher technique mention in Crisfield1991 and Crisfield1996
# TODO: implement dense output


class Riks:
    """Linear arc-length solver close to Riks method as dervied in Crisfield1991 
    section 9.3.2 p.273. A variable arc-length is chosen as shown by 
    Crisfield1981 or Crisfield 1983. For the first predictor a tangent predictor 
    is used. For all other predictors a simple secant predictor is used. This 
    enables the solver to 'run forward' instead of 'doubling back on its track'.

    References
    ----------
    - Wempner1971: https://doi.org/10.1016/0020-7683(71)90038-2 \\
    - Riks1972: https://doi.org/10.1115/1.3422829 \\
    - Riks1979: https://doi.org/10.1016/0020-7683(79)90081-7 \\
    - Crsfield1981: https://doi.org/10.1016/0045-7949(81)90108-5 \\
    - Crisfield1991: http://freeit.free.fr/Finite%20Element/Crisfield%20M.A.%20Vol.1.%20Non-Linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Essentials%20(Wiley,19.pdf \\
    - Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    - Neto1999: https://doi.org/10.1016/S0045-7825(99)00042-0
    """

    def a(self, x):
        """The most simple arc length equation restricting the change of all
        generalized coordinates w.r.t. the last converged Newton step."""
        nq = self.nq
        dq = x[:nq] - self.xk[:nq]
        return dq @ dq

    def a_x(self, x):
        nq = self.nq
        dq = x[:nq] - self.xk[:nq]
        return 2 * dq, np.zeros(self.nla), 0

    def R(self, x):
        return next(self.gen_analytic(x))

    def gen_numeric(self, x):
        yield self.R(x)
        yield approx_fprime(x, self.R, method="2-point")

    def gen_analytic(self, x):
        # extract generalized coordinates and Lagrange multipliers
        nq = self.nq
        nla = self.nla
        q = x[:nq]
        la = x[nq : nq + nla]
        la_arc = x[nq + nla]

        # evaluate all functions with t = la_arc -> model does not change!
        # - this requires the external force that should be scaled to be of the form
        #   F_ext(t, q) = t * F(q)
        # - the constraints for displacement control have to be of the form
        #   g(t, q) = t * g(q)
        t = la_arc
        h = self.model.h(t, q, self.u0)
        g = self.model.g(t, q)
        W_g = self.model.W_g(t, q)

        # build residual
        R = np.zeros(self.nx)
        R[:nq] = h + W_g @ la
        R[nq : nq + nla] = g
        R[nq + nla] = self.a(x) - self.ds**2

        yield R

        # evaluate all functions with t = la_arc -> model does not change!
        # - this requires the external force that should be scaled to be of the form
        #   F_ext(t, q) = t * F(q)
        # - the constraints for displacement control have to be of the form
        #   g(t, q) = t * g(q)
        h_q = self.model.h_q(t, q, self.u0)
        Wla_g_q = self.model.Wla_g_q(t, q, la)
        g_q = self.model.g_q(t, q)
        Rq_q = h_q + Wla_g_q

        # TODO: this is a hack and can't be fixed without specifying the scaled equations
        # but it is only two addition evaluations of the h vector and the constraints
        eps = 1.0e-6
        f_arc = (self.model.h(t + eps, q, self.u0) - h) / eps
        g_arc = (self.model.g(t + eps, q) - g) / eps

        # derivative of the arc length equation
        a_q, a_la, a_la_arc = self.a_x(x)

        yield bmat(
            [
                [Rq_q, W_g, f_arc[:, None]],
                [g_q, None, g_arc[:, None]],
                [a_q, a_la, a_la_arc],
            ],
            format="csr",
        )

    def __init__(
        self,
        model,
        tol=1e-10,
        max_newton_iter=50,
        iter_goal=4,
        la_arc0=1.0e-3,
        la_arc_span=np.array([1.0, 1.0]),
        numerical_jacobian=False,
        scale_exponent=0.5,
        debug=0,
    ):
        self.tol = tol
        self.max_newton_iter = max_newton_iter
        self.model = model
        self.iter_goal = iter_goal
        self.la_arc0 = la_arc0
        self.la_arc_span = la_arc_span

        self.newton_error_function = lambda R: np.absolute(R).max()

        if numerical_jacobian:
            self.gen = self.gen_numeric
        else:
            self.gen = self.gen_analytic

        # # check if we have an external force that is scaled by scalar parameter
        # if not len(model._Model__f_scaled_contr):
        #     raise RuntimeError('No scaled external force is given.')

        # parameter for the step size scaling
        self.MIN_FACTOR = 0.25  # minimal scaling factor
        self.MAX_FACTOR = 1.5  # maximal scaling factor

        # dimensions
        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla = self.model.nla_g
        self.nx = self.nq + self.nla + 1

        # initial
        self.q0 = self.model.q0
        self.la_g0 = self.model.la_g0
        self.la_arc0 = la_arc0
        self.u0 = np.zeros(model.nu)  # statics

        # initial values for generalized coordinates, lagrange multipliers and force scaling
        self.x0 = self.xk = np.concatenate(
            (self.q0, self.la_g0, np.array([self.la_arc0]))
        )

        ####################################################################################################
        # Solve linearized system for fixed external force using Newtons method.
        # From this solution we can extract the initial ds using the arc length equation.
        # All other ds values will be modified according to the number of used Newton steps,
        # see https://scicomp.stackexchange.com/questions/28137/initialize-arc-length-control-in-riks-method
        ####################################################################################################
        print(f"solve linear system using the initial arc length parameter")
        self.ds = 0  # initial ds has to be zero!
        xk1 = self.x0.copy()  # this copy is essential!
        gen = self.gen(xk1)
        R = next(gen)[:-1]
        error = self.newton_error_function(R)
        newton_iter = 0
        print(f"   * iter = {newton_iter}, error = {error:2.4e}")
        while error > self.tol:
            newton_iter += 1
            R_x = next(gen)[:-1, :-1]
            xk1[:-1] -= spsolve(R_x, R)
            gen = self.gen(xk1)
            R = next(gen)[:-1]
            error = self.newton_error_function(R)
            print(f"   * iter = {newton_iter}, error = {error:2.4e}")

        # compute initial ds from arc-length equation
        self.ds = self.a(xk1) ** 0.5
        if self.ds <= 0:
            raise RuntimeError("ds <= 0")
        print(
            f" => Newton converged in {newton_iter} iterations with error {error:2.4e}; initial ds: {self.ds:2.4e}"
        )

        # chose scaling exponent, see https://scicomp.stackexchange.com/questions/28137/initialize-arc-length-control-in-riks-method
        self.scale_ds = False
        if scale_exponent is not None:
            self.scale_ds = True
            self.scale_exponent = scale_exponent

        # debug information or not
        self.debug = debug

    def solve(self):
        # count number of force increments to get first increment with tangential predictor
        i = 0

        # extract number of generalized coordinates and number of Lagrange multipliers
        nq = self.nq
        nla = self.nla

        # initialize current generalized coordinates, Lagrange multipliers and force scaling
        la_arc = [self.la_arc0]
        q = [self.q0]
        la_g = [self.la_g0]
        xk1 = np.concatenate((self.q0, self.la_g0, np.array([self.la_arc0])))

        # compute initial residual and error
        gen = self.gen(xk1)
        R = next(gen)
        error = self.newton_error_function(R)

        # loop over ranges of force scaling
        while xk1[-1] > self.la_arc_span[0] and xk1[-1] < self.la_arc_span[1]:
            i += 1
            # use secant predictor for all other force increments than the first one
            if i > 1:
                # secand predictor for all but the first newton iteration
                dx = self.xk - self.x0
                xk1 += dx

                # ###################################
                # # prediction of Feng (see Neto1998)
                # ##################################
                # # secant predictor
                # Dx = self.xk - self.x0

                # # tangent predictor
                # # gen = self.gen(xk1)
                # # R = next(gen)
                # R_x = next(gen)
                # dx = spsolve(R_x, R)

                # inner = dx @ Dx
                # sign_inner = sign(inner)

                # # update with correspinding sign
                # xk1[:-1] += sign_inner * Dx[:-1]
                # # xk1[:-1] -= sign_inner * dx[:-1]

            else:
                # TODO:
                # find out why it is essential to solve for the generalized coordinates
                # and Lagrange-multipliers but not for the external force scaling
                R_x = next(gen)
                dx = spsolve(R_x[:-1, :-1], R[:-1])
                xk1[:-1] -= dx

            # compute initial residual and error
            gen = self.gen(xk1)
            R = next(gen)
            error = self.newton_error_function(R)

            newton_iter = 0
            if self.debug > 0:
                print(f"   * iter = {newton_iter}, error = {error:2.4e}")
            while (error > self.tol) and (newton_iter <= self.max_newton_iter):
                # solve linear system of equations
                R_x = next(gen)
                dx = spsolve(R_x, R)
                xk1 -= dx
                newton_iter += 1

                # check for convergence
                gen = self.gen(xk1)
                R = next(gen)
                error = self.newton_error_function(R)
                if self.debug > 0:
                    print(f"   * iter = {newton_iter}, error = {error:2.4e}")

            if newton_iter >= self.max_newton_iter:
                print(
                    f" - internal Netwon not converged for lambda = {xk1[-1]:2.4e} with error {error:2.2e}."
                )
                print(f"error = {error}")
                break
            else:
                print(
                    f" - internal Netwon converged for lambda = {xk1[-1]:2.4e} with error {error:2.2e} in {newton_iter} steps."
                )

            # scale ds such that iter goal is satisfied
            # disable scaling if we have halfed the ds parameter before or after the first iteration which requires lots of iterations
            # see Crisfield1991, section 9.5 (9.40) or (9.41) for the square root scaling
            if self.scale_ds and newton_iter != 0 and i > 1:
                fac = (self.iter_goal / newton_iter) ** self.scale_exponent
                if self.debug > 1:
                    ds_old = self.ds
                self.ds *= min(self.MAX_FACTOR, max(self.MIN_FACTOR, fac))
                if self.debug > 1:
                    print(f"   * ds: {ds_old} => {self.ds}")

            # store last converged newton step
            self.x0 = self.xk.copy()

            # store new converged newton step
            self.xk = xk1.copy()

            # append solutions to lists
            # these copies are essential!
            q.append(xk1[:nq].copy())
            la_g.append(xk1[nq : nq + nla].copy())
            la_arc.append(xk1[nq + nla].copy())

        # return solution object
        return Solution(t=np.asarray(la_arc), q=np.asarray(q), la_g=np.asarray(la_g))