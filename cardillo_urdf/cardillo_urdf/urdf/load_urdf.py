import numpy as np
import trimesh
from cardillo import System
from cardillo_urdf.parser import URDF
from cardillo_urdf.urdf import link_forward_kinematics
from cardillo.discrete import Meshed, Frame, RigidBody, RigidlyAttachedRigidBody
from cardillo.constraints import Revolute, RigidConnection, Prismatic
from cardillo.forces import Force
from cardillo.math import Spurrier, cross3, norm

def combine_transforms(transform1, transform2):
    """Combine two 4x4 homogeneous transformation matrices."""
    return np.dot(transform1, transform2)

def _is_massless(link):
    """Determine if a link is massless or inertialess."""
    return (
        link.inertial is None or 
        (hasattr(link.inertial, '_is_massless_connector') and link.inertial._is_massless_connector) or
        (hasattr(link.inertial, 'mass') and link.inertial.mass is None)
    )

def _create_frame(link, H_IS, H_CV):
    """Create a Frame object for a link, with optional visual mesh."""
    if len(link.visuals) != 0:
        mesh = trimesh.util.concatenate(
            link.visuals[0].geometry.meshes)
        return Meshed(Frame)(
            mesh_obj=mesh,
            B_r_CP=H_CV[link][:3, 3],
            A_BM=H_CV[link][:3, :3],
            r_OP=H_IS[link][:3, 3],
            A_IB=H_IS[link][:3, :3],
        )
    else:
        return Frame(
            r_OP=H_IS[link][:3, 3],
            A_IB=H_IS[link][:3, :3],
        )

def _create_rigid_body(link, H_IS, H_CV, q0, u0):
    """Create a RigidBody object for a link."""
    mass = link.inertial.mass if link.inertial else None
    inertia = link.inertial.inertia if link.inertial else None
    
    if len(link.visuals) != 0:
        mesh = trimesh.util.concatenate(
            link.visuals[0].geometry.meshes)
        return Meshed(RigidBody)(
            mesh_obj=mesh,
            B_r_CP=H_CV[link][:3, 3],
            A_BM=H_CV[link][:3, :3],
            mass=mass,
            B_Theta_C=inertia,
            q0=q0,
            u0=u0,
        )
    else:
        return RigidBody(
            mass=mass,
            B_Theta_C=inertia,
            q0=q0,
            u0=u0,
        )

def load_urdf(
    system,
    file,
    r_OC0=np.zeros(3),
    A_IC0=np.eye(3),
    v_C0=np.zeros(3),
    C0_Omega_0=np.zeros(3),
    initial_config=None,
    initial_vel=None,
    base_link_is_floating=False,
    gravitational_acceleration=None,
):
    """
    Load a URDF file into a Cardillo system with advanced tree-based processing.
    
    Args:
        system (System): Cardillo system to add robot to
        file (str): Path to URDF file
        r_OC0 (np.ndarray): Initial position of base link
        A_IC0 (np.ndarray): Initial orientation of base link
        v_C0 (np.ndarray): Initial linear velocity of base link
        C0_Omega_0 (np.ndarray): Initial angular velocity of base link
        initial_config (dict, optional): Initial joint configurations
        initial_vel (dict, optional): Initial joint velocities
        base_link_is_floating (bool, optional): Whether base link can move freely
        gravitational_acceleration (np.ndarray, optional): Gravity vector
    """
    # Load URDF and calculate kinematics
    urdf_system = URDF.load(file)
    H_IS, H_IL, H_IJ, H_SV, v_S, S_Omega = link_forward_kinematics(
        urdf_system,
        r_OC0=r_OC0,
        A_IC0=A_IC0,
        v_C0=v_C0,
        C0_Omega_0=C0_Omega_0,
        cfg=initial_config,
        vel=initial_vel,
    )
    initial_config = urdf_system._process_cfg(initial_config)
    
    # Prepare link tree for recursive processing
    link_tree = {}
    for joint in urdf_system.joints:
        parent = joint.parent
        child = joint.child
        if parent not in link_tree:
            link_tree[parent] = []
        link_tree[parent].append((child, joint))
    
    def process_link_branch(parent_name, parent_body=None, parent_joint=None):
        """Recursively process links and joints in the kinematic tree."""
        if parent_name not in link_tree:
            return
        
        for child_name, joint in link_tree[parent_name]:
            child_link = urdf_system.link_map[child_name]
            is_massless = _is_massless(child_link)
            
            # Print mass information
            mass = child_link.inertial.mass if child_link.inertial else None
            print(f"Link '{child_name}' Mass: {mass}")
            
            if is_massless and joint.joint_type == "fixed":
                # For massless links with fixed joints, redirect connections
                print("None - Massless fixed link")
                
                # Redirect children of massless link to its parent
                for grandchild_joint in [j for j in urdf_system.joints if j.parent == child_name]:
                    grandchild_joint.origin = combine_transforms(joint.origin, grandchild_joint.origin)
                    grandchild_joint.parent = parent_name
                    process_link_branch(grandchild_joint.child, parent_body, grandchild_joint)
            else:
                # Compute link transformation and initial conditions
                q0 = np.hstack([H_IS[child_link][:3, 3], Spurrier(H_IS[child_link][:3, :3])])
                u0 = np.hstack([v_S[child_link], S_Omega[child_link]])
                
                # Create cardillo body
                child_body = (
                    _create_rigid_body(child_link, H_IS, H_SV, q0, u0) 
                    if not is_massless 
                    else _create_frame(child_link, H_IS, H_SV)
                )
                
                child_body.name = child_link.name
                system.add(child_body)
                
                # Add gravity if applicable
                if gravitational_acceleration is not None and mass is not None:
                    grav = Force(mass * gravitational_acceleration, child_body)
                    grav.name = f"gravity_{child_body.name}"
                    system.add(grav)
                
                # Add joint connecting to parent
                if parent_joint and parent_joint.joint_type != "floating":
                    A_IB_child = H_IL[child_link][:3, :3]
                    r_OB_child = H_IL[child_link][:3, 3]
                    
                    if parent_joint.joint_type == "fixed":
                        # Use RigidlyAttachedRigidBody for fixed joints
                        rigid_attach = RigidlyAttachedRigidBody(
                            mass=mass,
                            B_Theta_C=child_link.inertial.inertia if child_link.inertial else None,
                            body=parent_body,
                            r_OC0=r_OB_child,
                            A_IB0=A_IB_child
                        )
                        rigid_attach.name = parent_joint.name
                        system.add(rigid_attach)
                    elif parent_joint.joint_type in ["revolute", "continuous", "prismatic"]:
                        # Revolute and prismatic joints
                        axis = parent_joint.axis / norm(parent_joint.axis)
                        
                        if parent_joint.joint_type in ["revolute", "continuous"]:
                            joint_constraint = Revolute(
                                parent_body,
                                child_body,
                                axis=0,
                                angle0=initial_config.get(parent_joint, 0),
                                r_OJ0=r_OB_child,
                                A_IJ0=A_IB_child
                            )
                        else:  # prismatic
                            joint_constraint = Prismatic(
                                parent_body,
                                child_body,
                                axis=0,
                                r_OJ0=r_OB_child,
                                A_IJ0=A_IB_child
                            )
                        
                        joint_constraint.name = parent_joint.name
                        system.add(joint_constraint)
                
                process_link_branch(child_name, child_body, parent_joint)
    
    # Process the entire kinematic tree starting from base link
    process_link_branch(urdf_system.base_link.name)
    
    system.assemble()
    return urdf_system