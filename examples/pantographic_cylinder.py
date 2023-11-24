from cardillo.math import A_IK_basic
from cardillo.rods import (
    RectangularCrossSection,
    Simo1986,
)
from cardillo.constraints import RigidConnection, Revolute

from cardillo.rods import K_R12_PetrovGalerkin_Quaternion as Rod

# from cardillo.beams import K_R12_PetrovGalerkin_AxisAngle as Rod

from cardillo import System

from cardillo.forces import LinearSpring, PDRotational
from cardillo.solver import Newton
from cardillo.discrete import Frame
from cardillo.rods._fitting import fit_configuration

from cardillo.utility import Export

import numpy as np
import pickle

if __name__ == "__main__":
    # number of elements
    nelements = 40

    # used polynomial degree
    polynomial_degree = 2
    basis = "Lagrange"

    scale = 1.0e0

    # Young's and shear modulus
    E = 1.7  # GPa
    G = E / (2 + 0.8)

    # length of the rod
    L = 5.0e2

    # slenderness and corresponding absolute tolerance for Newton-Raphson solver
    slenderness = 1.0e4 / 2
    atol = 1.0e-10

    # used cross section
    width_z = 3.4 * scale
    width_r = 2.8 * scale

    # cross section
    line_density = 1
    cross_section = RectangularCrossSection(line_density, width_z, width_r)
    A_rho0 = line_density * cross_section.area
    K_S_rho0 = line_density * cross_section.first_moment
    K_I_rho0 = line_density * cross_section.second_moment
    A = cross_section.area

    # quadratic beam material
    Ip, I2, I3 = np.diag(cross_section.second_moment)
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # ccw rod
    Q0 = Rod.straight_configuration(
        polynomial_degree,
        polynomial_degree,
        basis,
        basis,
        nelements,
        L,
    )
    rod = Rod(
        cross_section,
        material_model,
        A_rho0,
        K_S_rho0,
        K_I_rho0,
        polynomial_degree,
        polynomial_degree,
        nelements,
        Q=Q0,
        q0=Q0,
        basis_r=basis,
        basis_psi=basis,
    )

    # pivot
    hp = -width_r  # pivot height

    # helix
    n_coils = 2  # number of helix coils
    RO = (35.0 - width_r / 2) * scale  # helix outer radius
    RI = (RO - hp - width_r) * scale  # helix inner radius
    h = 600.0 * scale  # helix height
    # h = RO * 2 * np.pi * n_coils
    cO = h / (RO * 2 * np.pi * n_coils)
    LO = np.sqrt(1 + cO**2) * RO * 2 * np.pi * n_coils
    cI = h / (RI * 2 * np.pi * n_coils)
    LI = np.sqrt(1 + cI**2) * RI * 2 * np.pi * n_coils
    print(f"R0: {RO}")
    print(f"h: {h}")
    # print(f"c: {c}")
    print(f"n: {n_coils}")
    # print(f"L: {L}")

    # joint_type = "revolute"
    joint_type = "rigid"
    # joint_type = "no_joint"
    # joint_type = "spring"
    # test = "compression"
    # test = "extension"
    test = "torsion"
    # test = "shear"

    Gp = G
    rp = 2.0
    Jp = np.pi * rp**4 / 2

    # k = Gp * Jp / hp

    # # reference solution
    # def r(xi, R=RO, phi0=0.0, alpha=1, c=cO):
    #     alpha = alpha * 2 * np.pi * n_coils * xi
    #     return R * np.array(
    #         [np.sin(alpha + phi0), -np.cos(alpha + phi0), alpha * c * alpha]
    #     )

    # def A_IK(xi, phi0=0.0, alpha=1, c=cO):
    #     alpha = alpha * 2 * np.pi * n_coils * xi
    #     sa = np.sin(alpha + phi0)
    #     ca = np.cos(alpha + phi0)

    #     # # strange directors
    #     # e_x = np.array([alpha * ca, alpha * sa, c]) / np.sqrt(1 + c**2)
    #     # e_y = np.array([-sa, ca, 0])
    #     # e_z = np.array([-c * ca, -c * sa, alpha]) / np.sqrt(1 + c**2)

    #     # e_x = alpha * np.array([ca, sa, alpha * c]) / np.sqrt(1 + c**2)
    #     # e_y = alpha * np.array([-sa, ca, 0])
    #     # e_z = np.array([-c * alpha * ca, -alpha * c * sa, 1]) / np.sqrt(1 + c**2)

    #     # correct directors
    #     e_x = alpha * np.array([ca, sa, alpha * c]) / np.sqrt(1 + c**2)
    #     e_y = np.array([-c * ca, -c * sa, alpha]) / np.sqrt(1 + c**2)
    #     e_z = -np.array([-sa, ca, 0])

    #     return np.vstack((e_x, e_y, e_z)).T

    # reference solution
    def r(xi, R=RO, phi0=0.0, alpha=1, c=cO):
        Delta_phi = 2 * np.pi * n_coils * xi
        return R * np.array(
            [
                np.sin(alpha * Delta_phi + phi0),
                -np.cos(alpha * Delta_phi + phi0),
                c * Delta_phi,
            ]
        )

    # Serret-Frenet basis
    def A_IK(xi, phi0=0.0, alpha=1, c=cO):
        Delta_phi = 2 * np.pi * n_coils * xi
        sa = np.sin(alpha * Delta_phi + phi0)
        ca = np.cos(alpha * Delta_phi + phi0)

        e_x = np.array([alpha * ca, alpha * sa, c]) / np.sqrt(1 + c**2)
        e_y = np.array([-sa, ca, 0])
        e_z = np.array([-c * ca, -c * sa, alpha]) / np.sqrt(1 + c**2)

        return np.vstack((e_x, e_y, e_z)).T

    nxi = 81
    xis = np.linspace(0, 1, num=nxi)

    # individual rods
    n_rod = 10  # number of rods per layer
    Q0_list = []
    rod_list = []
    rod_ccw_list = []
    joint_list = []

    # load config and sol
    # load_config = True
    load_config = False
    load_sol = False
    from pathlib import Path
    import copy

    path = Path(__file__)
    folder = Path(
        path.parent, "results", path.stem, "torsion_ccw_2_coils_el=40", "hp=%s" % hp
    )
    Path(folder).mkdir(parents=True, exist_ok=True)
    filename = Path(folder, "initial_config")
    if load_config:
        Q0_list = pickle.load(open(filename, "rb"))
        for n in range(n_rod):
            rod_ccw = copy.deepcopy(rod)
            rod_ccw.q0 = Q0_list[2 * n].copy()
            rod_ccw.set_initial_strains(rod_ccw.q0)
            rod_ccw_list.append(rod_ccw)

            rod_cw = copy.deepcopy(rod)
            rod_cw.q0 = Q0_list[2 * n + 1].copy()
            rod_cw.set_initial_strains(rod_cw.q0)

            rod_list.extend((rod_ccw, rod_cw))
    else:
        for n in range(n_rod):
            rod_ccw = copy.deepcopy(rod)
            phi0 = 2 * np.pi * n / n_rod
            r_OP_ccw = np.array([r(xi, R=RO, alpha=1, c=cO, phi0=phi0) for xi in xis])
            A_IK_ccw = np.array([A_IK(xi, alpha=1, phi0=phi0) for xi in xis])
            Q0_ccw = fit_configuration(rod_ccw, r_OP_ccw, A_IK_ccw)
            rod_ccw.q0 = Q0_ccw.copy()
            Q0_list.append(Q0_ccw)
            rod_list.append(rod_ccw)
            rod_ccw_list.append(rod_ccw)

            rod_cw = copy.deepcopy(rod)
            r_OP_cw = np.array([r(xi, R=RI, alpha=-1, c=cI, phi0=phi0) for xi in xis])
            A_IK_cw = np.array([A_IK(xi, alpha=-1, c=cI, phi0=phi0) for xi in xis])
            Q0_cw = fit_configuration(rod_cw, r_OP_cw, A_IK_cw)
            Q0_list.append(Q0_cw)
            rod_cw.q0 = Q0_cw.copy()
            rod_list.append(rod_cw)

        file = open(filename, "wb")
        pickle.dump(Q0_list, file)
        file.close()

    # joints between ccw and cw rods
    revolute_joint_list = []
    for n in range(n_rod):
        phi0 = 2 * np.pi * n / n_rod
        rod_ccw = rod_list[2 * n]
        for nn in range(n_rod):
            for nc in range(n_coils * 2):
                dphi0 = (np.pi * (nn - n) / n_rod + nc * np.pi) % (2 * np.pi * n_coils)
                xi = dphi0 / (2 * np.pi * n_coils)
                if 0 < xi < 1:
                    r_OP_joint = r(xi, phi0=phi0)
                    A_IK_ccw = A_IK(xi, phi0=phi0)
                    frame_ID_ccw = (xi,)
                    frame_ID_cw = frame_ID_ccw
                    rod_cw = rod_list[2 * nn + 1]
                    if joint_type == "revolute":
                        joint = Revolute(
                            rod_ccw,
                            rod_cw,
                            axis=2,
                            r_OB0=r_OP_joint,
                            # A_IB0=A_IK_ccw,
                            frame_ID1=frame_ID_ccw,
                            frame_ID2=frame_ID_cw,
                        )
                        revolute_joint_list.append(joint)

                    elif joint_type == "rigid":
                        joint = RigidConnection(
                            rod_ccw,
                            rod_cw,
                            frame_ID1=frame_ID_ccw,
                            frame_ID2=frame_ID_cw,
                        )
                        revolute_joint_list.append(joint)

                    elif joint_type == "spring":
                        joint = PDRotational(Revolute, Spring=LinearSpring)(
                            subsystem1=rod_ccw,
                            subsystem2=rod_cw,
                            axis=2,
                            r_OB0=r_OP_joint,
                            # A_IB0=A_IK0,
                            k=k,
                            frame_ID1=frame_ID_ccw,
                            frame_ID2=frame_ID_cw,
                        )
                        revolute_joint_list.append(joint)

    # test plot
    # ax = plt.axes(projection="3d")
    # for n in range(n_rod):
    #     phi0 = 2 * np.pi * n / n_rod
    #     r_OP_ccw = np.array([r(xi, R=RO, alpha=1, c=cO, phi0=phi0) for xi in xis])
    #     A_IK_ccw = np.array([A_IK(xi, alpha=1, phi0=phi0) for xi in xis])
    #     r_OP_cw = np.array([r(xi, R=RI, alpha=-1, c=cI, phi0=phi0) for xi in xis])
    #     A_IK_cw = np.array([A_IK(xi, alpha=-1, phi0=phi0, c=cI) for xi in xis])
    #     # for i,xi in enumerate(xis):
    #     ax.plot3D(*r_OP_ccw.T, color="k")
    #     # ax.quiver(*r_OP_ccw[0].T, *A_IK_ccw[0].T[0],color='r')
    #     # ax.quiver(*r_OP_ccw[0].T, *A_IK_ccw[0].T[1],color='b')
    #     # ax.quiver(*r_OP_ccw[0].T, *A_IK_ccw[0].T[2],color='g')

    #     ax.plot3D(*r_OP_cw.T, color="b")
    #     # ax.quiver(*r_OP_cw[0].T, *A_IK_cw[0].T[0],color='r')
    #     # ax.quiver(*r_OP_cw[0].T, *A_IK_cw[0].T[1],color='b')
    #     # ax.quiver(*r_OP_cw[0].T, *A_IK_cw[0].T[2],color='g')
    # for n in range(n_rod):
    #     phi0 = 2 * np.pi * n / n_rod
    #     for nn in range(n_rod):
    #         for nc in range(n_coils * 2):
    #             # phi0 = 2 * np.pi * n / n_rod
    #             dphi0 = (np.pi * (nn - n) / n_rod + nc * np.pi) % (2 * np.pi)
    #             xi = dphi0 / (2 * np.pi * n_coils)
    #             if 0 < xi < 1:
    #                 r_OP_joint = r(xi, phi0=phi0)
    #                 A_IK_ccw = A_IK(xi, phi0=phi0)
    #                 ax.quiver(*r_OP_joint, *A_IK_ccw.T[2], color="r")
    #                 plt.show()
    #                 print(n,nn,nc,xi)
    #                 print("\n")

    # ax.set_xlim3d(0, 15)
    # ax.set_ylim3d(0, 15)
    # ax.set_zlim3d(0, 15)

    # joints between frames and rods
    system = System()
    Z_max = r(1)[-1]
    folder_test = Path(folder, test + "_" + joint_type)
    Path(folder_test).mkdir(exist_ok=True)
    r_OP_top = lambda t: np.array([0, 0, Z_max])
    A_IK_top = lambda t: np.eye(3)
    if test == "compression":
        r_OP_top = lambda t: np.array([0, 0, Z_max - h / 3 * t])
    elif test == "extension":
        r_OP_top = lambda t: np.array([0, 0, Z_max + h / 3 * t])
    elif test == "shear":
        r_OP_top = lambda t: np.array([0, 2 * RO * t, Z_max])
    elif test == "torsion":
        A_IK_top = lambda t: A_IK_basic(t * np.pi / 4).z()

    frame_top = Frame(r_OP=r_OP_top, A_IK=A_IK_top)

    for rod in rod_list:
        joint_bottom = RigidConnection(system.origin, rod, frame_ID2=(0,))
        joint_top = RigidConnection(frame_top, rod, frame_ID2=(1,))
        joint_list.extend((joint_bottom, joint_top))

    # assemble the system
    system.add(*rod_list)
    system.add(*joint_list)
    system.add(frame_top)
    system.add(*revolute_joint_list)
    # system.add(moment)
    system.assemble()

    # solve static system
    if load_sol == False:
        n_load_steps = 30
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            atol=atol,
            max_iter=30,
        )
        sol = solver.solve()
        filename = Path(folder_test, "sol")
        file_sol = open(filename, "wb")
        pickle.dump(sol, file_sol)
    elif load_sol:
        filename = Path(folder_test, "sol")
        sol = pickle.load(open(filename, "rb"))

    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    sol.t = t

    # t = [0, 1]
    # q = [q0, q0]

    ###########
    # animation
    ###########
    # animate_beam(t, q, rod_list, 2 * scale, show=True)

    ###########
    # export
    ###########
    # import cProfile, pstats, io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()

    # export centerline with directors and solution
    size = (nt, 3, 101)
    centerlines = np.zeros((2 * n_rod, 4, nt, 3, 101))
    file_centerlines = open(Path(folder_test, "centerlines"), "wb")
    # for i, rod in enumerate(rod_list):
    #     r_OP = np.zeros([nt, 3, 101])
    #     d1 = np.zeros([nt, 3, 101])
    #     d2 = np.zeros([nt, 3, 101])
    #     d3 = np.zeros([nt, 3, 101])
    #     for ti in range(nt):
    #         r_OP[ti], d1[ti], d2[ti], d3[ti] = rod.frames(sol.q[ti], 101)
    #     centerlines[i] = r_OP, d1, d2, d3
    # pickle.dump(centerlines, file_centerlines)

    # export total internal energy
    size = (nt, 3)
    E_pot = np.zeros((2 * n_rod, nt))
    file_E_pot = open(Path(folder_test, "total_energy"), "wb")
    for i, rod in enumerate(rod_list):
        for ti in range(nt):
            E_pot[i, ti] = rod.E_pot(ti, sol.q[ti, rod.qDOF])

    pickle.dump(E_pot, file_E_pot)

    folder_vtk = Path(folder_test, "vtk_files")
    e = Export(
        path=folder_vtk.parent,
        folder_name=folder_vtk.stem,
        overwrite=True,
        fps=nt,
        solution=sol,
        system=system,
    )
    e.export_contr(rod_list[0], level="centerline + directors", num=100)
    for rod in rod_list:
        # e.export_contr(rod, level="centerline + directors", num=100)
        e.export_contr(rod, level="volume")

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
