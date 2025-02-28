import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
from collections import OrderedDict
import six

# -------------------- Helper Classes --------------------
class Inertial:
    def __init__(self, mass=None, origin=None, inertia=None):
        self.mass = mass
        self.origin = origin
        self.inertia = inertia

class Visual:
    def __init__(self, geometry=None, material=None, origin=None):
        self.geometry = geometry
        self.material = material
        self.origin = origin

class Collision:
    def __init__(self, geometry=None, origin=None):
        self.geometry = geometry
        self.origin = origin

class Link:
    def __init__(self, name, inertial=None, visuals=None, collisions=None):
        self.name = name
        self.inertial = inertial
        self.visuals = visuals if visuals else []
        self.collisions = collisions if collisions else []

class Joint:
    def __init__(self, name, joint_type, parent, child, origin=None, axis=None):
        self.name = name
        self.type = joint_type
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis

# -------------------- URDF Class --------------------
class URDF:
    def __init__(
        self,
        name,
        file,
        links,
        joints=None,
        transmissions=None,
        materials=None,
        other_xml=None,
    ):
        if joints is None:
            joints = []
        if transmissions is None:
            transmissions = []
        if materials is None:
            materials = []
        
        self.file = file
        self.links = links
        self.name = name
        self.other_xml = other_xml
        self.mesh_need_to_mirror = []

        # No setters for these
        self._links = list(links)
        self._joints = list(joints)
        self._transmissions = list(transmissions)
        self._materials = list(materials)

        # Set up private helper maps from name to value
        self._link_map = {}
        self._joint_map = {}
        self._transmission_map = {}
        self._material_map = {}

        for x in self._links:
            if x.name in self._link_map:
                raise ValueError("Two links with name {} found".format(x.name))
            self._link_map[x.name] = x

        for x in self._joints:
            if x.name in self._joint_map:
                raise ValueError("Two joints with name {} " "found".format(x.name))
            self._joint_map[x.name] = x

        for x in self._transmissions:
            if x.name in self._transmission_map:
                raise ValueError(
                    "Two transmissions with name {} " "found".format(x.name)
                )
            self._transmission_map[x.name] = x

        for x in self._materials:
            if x.name in self._material_map:
                raise ValueError("Two materials with name {} " "found".format(x.name))
            self._material_map[x.name] = x

        # Synchronize materials between links and top-level set
        self._merge_materials()

        # Validate the joints and transmissions
        actuated_joints = self._validate_joints()
        self._validate_transmissions()

        # Create the link graph and base link/end link sets
        self._G = nx.DiGraph()

        # Add all links
        for link in self._links:
            self._G.add_node(link)

        # Add all edges from CHILDREN TO PARENTS, with joints as their object
        for joint in self._joints:
            parent = self._link_map[joint.parent]
            child = self._link_map[joint.child]
            self._G.add_edge(child, parent, joint=joint)

        # Validate the graph and get the base and end links
        self._base_link, self._end_links = self._validate_graph()

        # Cache the paths to the base link
        self._paths_to_base = nx.shortest_path(self._G, target=self._base_link)

        self._actuated_joints = self._sort_joints(actuated_joints)

        # Cache the reverse topological order (useful for speeding up FK,
        # as we want to start at the base and work outward to cache
        # computation.
        self._reverse_topo = list(reversed(list(nx.topological_sort(self._G))))

    def _validate_joints(self):
        """Raise an exception of any joints are invalidly specified.

        Checks for the following:

        - Joint parents are valid link names.
        - Joint children are valid link names that aren't the same as parent.
        - Joint mimics have valid joint names that aren't the same joint.

        Returns
        -------
        actuated_joints : list of :class:`.Joint`
            The joints in the model that are independently controllable.
        """
        actuated_joints = []
        for joint in self._joints:
            if joint.parent not in self._link_map:
                raise ValueError(
                    "Joint {} has invalid parent link name {}".format(
                        joint.name, joint.parent
                    )
                )
            if joint.child not in self._link_map:
                raise ValueError(
                    "Joint {} has invalid child link name {}".format(
                        joint.name, joint.child
                    )
                )
            if joint.child == joint.parent:
                raise ValueError(
                    "Joint {} has matching parent and child".format(joint.name)
                )
            if joint.mimic is not None:
                if joint.mimic.joint not in self._joint_map:
                    raise ValueError(
                        "Joint {} has an invalid mimic joint name {}".format(
                            joint.name, joint.mimic.joint
                        )
                    )
                if joint.mimic.joint == joint.name:
                    raise ValueError(
                        "Joint {} set up to mimic itself".format(joint.mimic.joint)
                    )
            elif joint.joint_type != "fixed":
                actuated_joints.append(joint)

        # Do a depth-first search
        return actuated_joints
    
    def _validate_transmissions(self):
        """Raise an exception of any transmissions are invalidly specified.

        Checks for the following:

        - Transmission joints have valid joint names.
        """
        for t in self._transmissions:
            for joint in t.joints:
                if joint.name not in self._joint_map:
                    raise ValueError(
                        "Transmission {} has invalid joint name "
                        "{}".format(t.name, joint.name)
                    )
    
    def _merge_materials(self):
        """Merge the top-level material set with the link materials."""
        for link in self._links:
            for v in link.visuals:
                if v.material is None:
                    continue
                if v.material.name in self.material_map:
                    v.material = self._material_map[v.material.name]
                else:
                    self._materials.append(v.material)
                    self._material_map[v.material.name] = v.material

    def parse(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        # Parse links
        for link in root.findall('link'):
            name = link.get('name')
            inertial = self._parse_inertial(link.find('inertial'))
            visuals = self._parse_visuals(link.findall('visual'))
            collision = self._parse_collision(link.find('collision'))

            link_obj = Link(name, inertial, visuals, collision)
            self.links.append(link_obj)
            self.link_map[name] = link_obj

            if self.base_link is None:
                self.base_link = link_obj

        # Parse joints
        for joint in root.findall('joint'):
            name = joint.get('name')
            joint_type = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            origin = self._parse_origin(joint.find('origin'))
            axis = self._parse_axis(joint.find('axis'))

            joint_obj = Joint(name, joint_type, parent, child, origin, axis)
            self.joints.append(joint_obj)
            self.joint_map[name] = joint_obj

        # Build graph
        for link in self.links:
            self._G.add_node(link)
        for joint in self.joints:
            parent = self.link_map[joint.parent]
            child = self.link_map[joint.child]
            self._G.add_edge(child, parent, joint=joint)

    def _parse_inertial(self, inertial_elem):
        if inertial_elem is None:
            return None

        mass_elem = inertial_elem.find("mass")
        mass = 0.0
        if mass_elem is not None and mass_elem.text:
            try:
                mass = float(mass_elem.text)
            except ValueError:
                print(f"Warning: Invalid mass value for {inertial_elem}: {mass_elem.text}")

        inertia_elem = inertial_elem.find("inertia")
        inertia = None
        if inertia_elem is not None:
            inertia = np.array([
                float(inertia_elem.get("ixx", 0)),
                float(inertia_elem.get("ixy", 0)),
                float(inertia_elem.get("ixz", 0)),
                float(inertia_elem.get("iyy", 0)),
                float(inertia_elem.get("iyz", 0)),
                float(inertia_elem.get("izz", 0))
            ])

        origin_elem = inertial_elem.find("origin")
        origin = self._parse_origin(origin_elem)

        return Inertial(mass, origin, inertia)

    def _parse_visuals(self, visual_elems):
        visuals = []
        for visual_elem in visual_elems:
            geometry_elem = visual_elem.find("geometry")
            geometry = self._parse_geometry(geometry_elem)
            material_elem = visual_elem.find("material")
            material = material_elem.get("name") if material_elem is not None else None
            origin_elem = visual_elem.find("origin")
            origin = self._parse_origin(origin_elem)
            visuals.append(Visual(geometry, material, origin))
        return visuals

    def _parse_collision(self, collision_elem):
        if collision_elem is None:
            return None

        geometry_elem = collision_elem.find("geometry")
        geometry = self._parse_geometry(geometry_elem)
        origin_elem = collision_elem.find("origin")
        origin = self._parse_origin(origin_elem)

        return Collision(geometry, origin)

    def _parse_geometry(self, geometry_element):
        for box_element in geometry_element.iter('box'):
            size = list(map(float, box_element.get('size', '1 1 1').split()))
            return {'type': 'box', 'size': size}
        for sphere_element in geometry_element.iter('sphere'):
            radius = float(sphere_element.get('radius', '0'))
            return {'type': 'sphere', 'radius': radius}
        for cylinder_element in geometry_element.iter('cylinder'):
            radius = float(cylinder_element.get('radius', '0'))
            length = float(cylinder_element.get('length', '0'))
            return {'type': 'cylinder', 'radius': radius, 'length': length}
        return None

    def _parse_origin(self, origin_elem):
        if origin_elem is None:
            return np.eye(4)

        xyz = origin_elem.get('xyz', '0 0 0').split()
        rpy = origin_elem.get('rpy', '0 0 0').split()

        xyz = np.array(list(map(float, xyz)))
        rpy = np.array(list(map(float, rpy)))

        rotation = self.rpy_to_matrix(rpy)

        transformation = np.eye(4)
        transformation[:3, 3] = xyz
        transformation[:3, :3] = rotation

        return transformation

    def _parse_axis(self, axis_elem):
        if axis_elem is None:
            return None
        return np.array([float(val) for val in axis_elem.get("xyz", "0 0 0").split()], dtype=np.float64)

    def rpy_to_matrix(self, rpy):
        roll, pitch, yaw = rpy

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def link_fk(self, cfg=None, link=None, links=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        link : str or :class:`.Link`
            A single link or link name to return a pose for.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only these links will be in the returned map. If neither
            link nor links are specified all links are returned.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.

        Returns
        -------
        fk : dict or (4,4) float
            A map from links to 4x4 homogenous transform matrices that
            position them relative to the base link's frame, or a single
            4x4 matrix if ``link`` is specified.
        """
        # Process config value
        joint_cfg = self._process_cfg(cfg)

        # Process link set
        link_set = set()
        if link is not None:
            if isinstance(link, six.string_types):
                link_set.add(self._link_map[link])
            elif isinstance(link, Link):
                link_set.add(link)
        elif links is not None:
            for lnk in links:
                if isinstance(lnk, six.string_types):
                    link_set.add(self._link_map[lnk])
                elif isinstance(lnk, Link):
                    link_set.add(lnk)
                else:
                    raise TypeError(
                        "Got object of type {} in links list".format(type(lnk))
                    )
        else:
            link_set = self.links

        # Compute forward kinematics in reverse topological order
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            pose = np.eye(4, dtype=np.float64)
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfg:
                        cfg = joint_cfg[mimic_joint]
                        cfg = joint.mimic.multiplier * cfg + joint.mimic.offset
                elif joint in joint_cfg:
                    cfg = joint_cfg[joint]
                pose = joint.get_child_pose(cfg).dot(pose)

                # Check existing FK to see if we can exit early
                if parent in fk:
                    pose = fk[parent].dot(pose)
                    break
            fk[lnk] = pose

        if link:
            if isinstance(link, six.string_types):
                return fk[self._link_map[link]]
            else:
                return fk[link]
        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def link_fk_batch(self, cfgs=None, link=None, links=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = set()
        if link is not None:
            if isinstance(link, six.string_types):
                link_set.add(self._link_map[link])
            elif isinstance(link, Link):
                link_set.add(link)
        elif links is not None:
            for lnk in links:
                if isinstance(lnk, six.string_types):
                    link_set.add(self._link_map[lnk])
                elif isinstance(lnk, Link):
                    link_set.add(lnk)
                else:
                    raise TypeError(
                        "Got object of type {} in links list".format(type(lnk))
                    )
        else:
            link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = np.tile(np.eye(4, dtype=np.float64), (n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = (
                            joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                        )
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]
                poses = np.matmul(joint.get_child_poses(cfg_vals, n_cfgs), poses)

                if parent in fk:
                    poses = np.matmul(fk[parent], poses)
                    break
            fk[lnk] = poses

        if link:
            if isinstance(link, six.string_types):
                return fk[self._link_map[link]]
            else:
                return fk[link]
        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_geometry_fk(self, cfg=None, links=None):
        """Computes the poses of the URDF's visual geometries using fk. """
        lfk = self.link_fk(cfg=cfg, links=links)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = lfk[link].dot(visual.origin)
        return fk

    def visual_geometry_fk_batch(self, cfgs=None, links=None):
        """Computes the poses of the URDF's visual geometries using fk. """
        lfk = self.link_fk_batch(cfgs=cfgs, links=links)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = np.matmul(lfk[link], visual.origin)
        return fk


    def _compute_reverse_topo(self):
        visited = set()
        order = []

        def visit(link_name):
            if link_name in visited:
                return
            visited.add(link_name)
            for joint in self.joints:
                if joint.parent == link_name:
                    visit(joint.child)
            order.append(self.link_map[link_name])

        if self.base_link:
            visit(self.base_link.name)
        self._reverse_topo = list(reversed(order))

    def _validate_graph(self):
        if not nx.is_weakly_connected(self._G):
            link_clusters = []
            for cc in nx.weakly_connected_components(self._G):
                cluster = [n.name for n in cc]
                link_clusters.append(cluster)
            message = "Links are not all connected. Connected components are:"
            for lc in link_clusters:
                message += "\n\t" + " ".join(lc)
            raise ValueError(message)

        if not nx.is_directed_acyclic_graph(self._G):
            raise ValueError("There are cycles in the link graph")

        base_link = None
        end_links = []
        for n in self._G:
            if len(nx.descendants(self._G, n)) == 0:
                if base_link is None:
                    base_link = n
                else:
                    raise ValueError(f"Links {n.name} and {base_link.name} are both base links!")
            if len(nx.ancestors(self._G, n)) == 0:
                end_links.append(n)
        return base_link, end_links

    def _process_cfg(self, cfg):
        """Process a joint configuration spec into a dictionary mapping joints to configuration values."""
        joint_cfg = {}
        if cfg is None:
            return joint_cfg
        if isinstance(cfg, dict):
            for joint in cfg:
                if isinstance(joint, six.string_types):
                    joint_cfg[self.joint_map[joint]] = cfg[joint]
                elif isinstance(joint, Joint):
                    joint_cfg[joint] = cfg[joint]
        elif isinstance(cfg, (list, tuple, np.ndarray)):
            if len(cfg) != len(self.actuated_joints):
                raise ValueError(
                    "Cfg must have same length as actuated joints if specified as a numerical array"
                )
            for joint, value in zip(self.actuated_joints, cfg):
                joint_cfg[joint] = value
        else:
            raise TypeError("Invalid type for config")
        return joint_cfg

    def _process_cfgs(self, cfgs):
        joint_cfg = {j: [] for j in self.actuated_joints}
        n_cfgs = None
        if isinstance(cfgs, dict):
            for joint in cfgs:
                if isinstance(joint, six.string_types):
                    joint_cfg[self.joint_map[joint]] = cfgs[joint]
                else:
                    joint_cfg[joint] = cfgs[joint]
                if n_cfgs is None:
                    n_cfgs = len(cfgs[joint])
        elif isinstance(cfgs, (list, tuple, np.ndarray)):
            n_cfgs = len(cfgs)
            if isinstance(cfgs[0], dict):
                for cfg in cfgs:
                    for joint in cfg:
                        if isinstance(joint, six.string_types):
                            joint_cfg[self.joint_map[joint]].append(cfg[joint])
                        else:
                            joint_cfg[joint].append(cfg[joint])
            elif cfgs[0] is None:
                pass
            else:
                cfgs = np.asanyarray(cfgs, dtype=np.float64)
                for i, j in enumerate(self.actuated_joints):
                    joint_cfg[j] = cfgs[:, i]
        else:
            raise ValueError("Incorrectly formatted config array")

        for j in joint_cfg:
            if len(joint_cfg[j]) == 0:
                joint_cfg[j] = None
            elif len(joint_cfg[j]) != n_cfgs:
                raise ValueError("Inconsistent number of configurations for joints")

        return joint_cfg, n_cfgs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes):
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path, lazy_load_meshes)

        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs["other_xml"] = data
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        if self.other_xml:
            extra_tree = ET.fromstring(self.other_xml)
            for child in extra_tree:
                node.append(child)
        return node