"""Small deterministic mesh authoring helpers; metres, X forward, Z up."""

import numpy as np


class Mesh:
    def __init__(self):
        self.vertices, self.indices = [], []

    def add(self, vertices, faces, position=(0, 0, 0), rotation=None):
        points = np.asarray(vertices, float)
        if rotation is not None:
            points = points @ rotation.T
        start = len(self.vertices)
        self.vertices.extend((points + position).tolist())
        self.indices.extend((np.asarray(faces) + start).tolist())

    def box(self, position, size):
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * np.array(size) / 2
        # Separate face vertices retain hard edges in the browser's normal pass.
        for quad in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                     (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.add(vertices[list(quad)], [[0, 1, 2], [0, 2, 3]], position)

    def rounded_box(self, position, size, bevel):
        half = np.asarray(size) / 2
        core = half - bevel
        vertices, faces = [], []
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            grids = [np.array([-half[i], -half[i] + bevel * .3, -core[i],
                              core[i], half[i] - bevel * .3, half[i]]) for i in other]
            for sign in (-1, 1):
                start = len(vertices)
                for x in grids[0]:
                    for y in grids[1]:
                        p = np.zeros(3); p[axis] = sign * half[axis]
                        p[other] = [x, y]
                        nearest = np.clip(p, -core, core)
                        direction = p - nearest
                        vertices.append(nearest + direction / np.linalg.norm(direction) * bevel)
                for i in range(5):
                    for j in range(5):
                        a = start + i * 6 + j
                        quad = [a, a + 6, a + 7, a + 1]
                        if sign * (-1 if axis == 1 else 1) < 0:
                            quad.reverse()
                        faces.extend([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
        self.add(vertices, faces, position)

    def tube(self, a, b, radius, segments=16):
        a, b = np.array(a), np.array(b)
        axis = b - a
        length = np.linalg.norm(axis)
        axis /= length
        helper = np.array([0, 0, 1] if abs(axis[2]) < .9 else [1, 0, 0])
        u = np.cross(axis, helper); u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        ring = np.array([np.cos(t) * u + np.sin(t) * v for t in np.arange(segments) * 2 * np.pi / segments]) * radius
        vertices = np.vstack((ring, ring + axis * length, [0, 0, 0], axis * length))
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend([[i, j, j + segments], [i, j + segments, i + segments],
                          [2 * segments, j, i], [2 * segments + 1, i + segments, j + segments]])
        self.add(vertices, faces, a)

    def ring(self, position, radius, thickness):
        vertices, faces = [], []
        for i in range(40):
            a = i * 2 * np.pi / 40
            for j in range(8):
                b = j * 2 * np.pi / 8
                r = radius + thickness * np.cos(b)
                vertices.append([r * np.cos(a), r * np.sin(a), thickness * np.sin(b)])
        for i in range(40):
            for j in range(8):
                a, b = i * 8 + j, ((i + 1) % 40) * 8 + j
                c, d = ((i + 1) % 40) * 8 + (j + 1) % 8, i * 8 + (j + 1) % 8
                faces.extend([[a, b, c], [a, c, d]])
        self.add(vertices, faces, position)

    def propeller(self, radius, chord, count):
        # Tapered, swept blades with a shallow pitch; this is authored visual geometry.
        for i in range(count):
            angle = 2 * np.pi * i / count
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            outline = np.array([[radius * .12, -chord * .3], [radius * .9, -chord * .1],
                                [radius, chord * .25], [radius * .94, chord * .5],
                                [radius * .18, chord * .6]])
            vertices = [[x, y, y * .15 + z] for z in (-chord * .025, chord * .025) for x, y in outline]
            faces = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [5, 6, 7], [5, 7, 8], [5, 8, 9]]
            for j in range(5):
                k = (j + 1) % 5
                faces.extend([[j, k, k + 5], [j, k + 5, j + 5]])
            self.add(vertices, faces, rotation=rotation)

    def record(self, name, color, roughness, metalness, position=(0, 0, 0)):
        return dict(name=name, color=color, roughness=roughness, metalness=metalness,
                    position=list(position), vertices=np.round(self.vertices, 7).ravel().tolist(),
                    indices=np.asarray(self.indices).ravel().tolist())
