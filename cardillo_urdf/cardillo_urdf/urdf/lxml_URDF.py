import numpy as np
from lxml import etree

class URDF:
    def __init__(self, file_path):
        self.links = []
        self.joints = []
        self.base_link = None
        self._parse(file_path)

    def _parse(self, file_path):
        with open(file_path, 'r') as f:
            xml_string = f.read()
        root = etree.fromstring(bytes(xml_string, encoding='utf-8'))

        # Parse links
        for link_elem in root.findall(".//link"):
            link = self._parse_link(link_elem)
            self.links.append(link)

        # Parse joints
        for joint_elem in root.findall(".//joint"):
            joint = self._parse_joint(joint_elem)
            self.joints.append(joint)

        # Set the base link
        self.base_link = self.links[0] if self.links else None

    def _parse_link(self, link_elem):
        name = link_elem.get("name")
        inertial_elem = link_elem.find("inertial")
        inertial = self._parse_inertial(inertial_elem) if inertial_elem is not None else None
        visuals = [self._parse_visual(visual_elem) for visual_elem in link_elem.findall("visual")]
        return {
            'name': name,
            'inertial': inertial,
            'visuals': visuals
        }

    def _parse_inertial(self, inertial_elem):
        mass = float(inertial_elem.find("mass").get("value"))
        inertia_elem = inertial_elem.find("inertia")
        inertia = {
            'ixx': float(inertia_elem.get("ixx")),
            'iyy': float(inertia_elem.get("iyy")),
            'izz': float(inertia_elem.get("izz")),
            'ixy': float(inertia_elem.get("ixy")),
            'ixz': float(inertia_elem.get("ixz")),
            'iyz': float(inertia_elem.get("iyz"))
        }
        origin_elem = inertial_elem.find("origin")
        origin = self._parse_origin(origin_elem)
        return {
            'mass': mass,
            'inertia': inertia,
            'origin': origin
        }

    def _parse_visual(self, visual_elem):
        geometry_elem = visual_elem.find("geometry")
        mesh_elem = geometry_elem.find("mesh") if geometry_elem is not None else None
        mesh = {'filename': mesh_elem.get("filename")} if mesh_elem is not None else None
        origin_elem = visual_elem.find("origin")
        origin = self._parse_origin(origin_elem)
        return {
            'geometry': {
                'mesh': mesh
            },
            'origin': origin
        }

    def _parse_joint(self, joint_elem):
        name = joint_elem.get("name")
        joint_type = joint_elem.get("type")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        parent = parent_elem.get("link") if parent_elem is not None else None
        child = child_elem.get("link") if child_elem is not None else None
        origin_elem = joint_elem.find("origin")
        origin = self._parse_origin(origin_elem)
        axis_elem = joint_elem.find("axis")
        axis = (
            [float(a) for a in axis_elem.get("xyz").split()]
            if axis_elem is not None else [1.0, 0.0, 0.0]
        )
        return {
            'name': name,
            'joint_type': joint_type,
            'parent': parent,
            'child': child,
            'origin': origin,
            'axis': axis
        }

    def _parse_origin(self, origin_elem):
        if origin_elem is None:
            return {
                'xyz': [0.0, 0.0, 0.0],
                'rpy': [0.0, 0.0, 0.0],
            }
        xyz_attr = origin_elem.get("xyz")
        rpy_attr = origin_elem.get("rpy")
        xyz = [float(x) for x in (xyz_attr.split() if xyz_attr else "0 0 0".split())]
        rpy = [float(r) for r in (rpy_attr.split() if rpy_attr else "0 0 0".split())]
        return {'xyz': xyz, 'rpy': rpy}

    def _process_cfg(self, cfg):
        """Process joint configuration data.
        Args:
            cfg (dict): Dictionary containing configuration data.
        Returns:
            dict: Processed configuration data.
        """
        processed_cfg = {}
        for joint in self.joints:
            joint_name = joint['name']
            if joint_name in cfg:
                processed_cfg[joint_name] = {
                    'position': cfg[joint_name].get('position', 0.0),
                    'velocity': cfg[joint_name].get('velocity', 0.0),
                    'effort': cfg[joint_name].get('effort', 0.0),
                }
        return processed_cfg
