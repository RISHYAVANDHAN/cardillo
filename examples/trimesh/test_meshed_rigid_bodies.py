import trimesh
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.forces import Force
from cardillo.discrete import (
    RigidBody,
    Frame,
    Meshed,
    Box,
    Cone,
    Cylinder,
    Sphere,
    Capsule,
)
from cardillo.math import A_IK_basic, Spurrier
from cardillo.solver import BackwardEuler

from cardillo.visualization import Export

from cardillo.visualization.trimesh import show_system, animate_system

if __name__ == "__main__":
    path = Path(__file__)
    vtk_export = False

    MeshedFrame = Meshed(Frame)
    MeshedRB = Meshed(RigidBody)

    cube_mesh = trimesh.creation.box(extents=[3, 3, 3])
    plane_mesh = cube_mesh.copy().apply_transform(np.diag([1, 1, 0.001, 1]))
    frame = MeshedFrame(plane_mesh, K_r_SP=np.array([0, 0, 0]))

    q10 = np.concatenate([np.array([0, 0, 1]), Spurrier(A_IK_basic(np.pi / 4).x())])
    # box_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    # rigid_body1 = MeshedRB(box_mesh, density=2, mass=1, K_Theta_S=np.eye(3), q0=q10)
    rigid_body1 = Box(RigidBody)(
        dimensions=[0.2, 0.2, 0.1], density=2, mass=1, K_Theta_S=np.eye(3), q0=q10
    )

    q20 = np.concatenate([np.array([0, 1, 1]), Spurrier(A_IK_basic(np.pi / 2).x())])
    scale = 3
    tippedisk_mesh = trimesh.load_mesh(
        Path.joinpath(path.parent, "tippedisk.stl")
    ).apply_transform(np.diag([scale, scale, scale, 1]))
    rigid_body2 = MeshedRB(
        tippedisk_mesh, density=1, mass=1, K_Theta_S=np.eye(3), q0=q20
    )

    q30 = np.concatenate([np.array([1, 1, 1]), Spurrier(A_IK_basic(-np.pi / 4).x())])
    rigid_body3 = Cone(RigidBody)(
        radius=0.1, height=0.2, density=2, mass=1, K_Theta_S=np.eye(3), q0=q30
    )

    q40 = np.concatenate([np.array([1, -1, 1]), Spurrier(A_IK_basic(0).x())])
    rigid_body4 = Cylinder(RigidBody)(
        radius=0.1, height=0.2, density=2, mass=1, K_Theta_S=np.eye(3), q0=q40
    )

    q50 = np.concatenate([np.array([0, -1, 1]), np.array([1, 0, 0, 0])])
    rigid_body5 = Sphere(RigidBody)(
        radius=0.1, subdivisions=3, density=2, mass=1, K_Theta_S=np.eye(3), q0=q50
    )

    q60 = np.concatenate([np.array([1, 0, 1]), Spurrier(A_IK_basic(-np.pi / 3).x())])
    rigid_body6 = Capsule(RigidBody)(
        radius=0.1, height=0.2, density=2, mass=1, K_Theta_S=np.eye(3), q0=q60
    )

    system = System()
    system.add(frame)
    system.add(rigid_body1)
    system.add(rigid_body2)
    system.add(rigid_body3)
    system.add(rigid_body4)
    system.add(rigid_body5)
    system.add(rigid_body6)
    system.add(Force(np.array([0, 0, -10 * rigid_body1.mass]), rigid_body1))
    system.add(Force(np.array([0, 0, -10 * rigid_body2.mass]), rigid_body2))
    system.add(Force(np.array([0, 0, -10 * rigid_body3.mass]), rigid_body3))
    system.add(Force(np.array([0, 0, -10 * rigid_body4.mass]), rigid_body4))
    system.add(Force(np.array([0, 0, -10 * rigid_body5.mass]), rigid_body5))
    system.add(Force(np.array([0, 0, -10 * rigid_body6.mass]), rigid_body6))
    system.assemble()

    # show_system(system, system.t0, system.q0, origin_size=0.05) # this will end the execution of the file!!

    sol = BackwardEuler(system, 5, 1e-1).solve()

    if vtk_export:
        path = Path(__file__)
        e = Export(
            path=path.parent,
            folder_name=path.stem,
            overwrite=True,
            fps=30,
            solution=sol,
        )
        e.export_contr(system.origin)
        e.export_contr(frame)
        e.export_contr(rigid_body1)
        e.export_contr(rigid_body2)
        e.export_contr(rigid_body3)
        e.export_contr(rigid_body4)
        e.export_contr(rigid_body5)
        e.export_contr(rigid_body6)

    animate_system(system, sol.t, sol.q, t_factor=1, fps=10, origin_size=0.05)
