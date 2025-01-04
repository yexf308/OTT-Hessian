import jax
import numpy as np
import jax.numpy as jnp
import ott
from sklearn.datasets import make_blobs
import random

import scipy.spatial.distance
from scipy.stats import special_ortho_group


def sample_points_uniform(n, m, dim, seed, uniform=True):
    np.random.seed(seed)
    x = np.random.uniform(size=(n, dim))
    y = np.random.uniform(size=(m, dim))

    if uniform:
        a = np.ones(n) / n
        b = np.ones(m) / m
    else:
        a = np.random.uniform(size=(n,)) + 0.1
        b = np.random.uniform(size=(m,)) + 0.1
        a = a / np.sum(a) # marginal dist, need to be uniform for spectrum study
        b = b / np.sum(b)

    a_jx = jnp.array(a)
    b_jx = jnp.array(b)
    x_jx = jnp.array(x)
    y_jx = jnp.array(y)

    return a_jx, b_jx, x_jx, y_jx

def sample_points_gaussian(n, m, dim, seed, uniform=True):
    np.random.seed(seed)
    x = np.random.normal(size=(n, dim))
    y = np.random.normal(size=(m, dim))

    if uniform:
        a = np.ones(n) / n
        b = np.ones(m) / m
    else:
        a = np.random.uniform(size=(n,)) + 0.1
        b = np.random.uniform(size=(m,)) + 0.1
        a = a / np.sum(a) # marginal dist, need to be uniform for spectrum study
        b = b / np.sum(b)

    a_jx = jnp.array(a)
    b_jx = jnp.array(b)
    x_jx = jnp.array(x)
    y_jx = jnp.array(y)

    return a_jx, b_jx, x_jx, y_jx

def sample_bolb(n, d_X, d_Y, blob_std, noise, seed):
  np.random.seed(seed)
  x, data_mem  = make_blobs(n_samples=n, n_features=d_X, centers=len(blob_std), cluster_std=blob_std)
  mu           = np.ones((n,)) / n
  nv           = np.ones((n,)) / n
  w            = np.random.normal(size=[d_X,d_Y])

  y            = x.dot(w) + noise*np.random.normal(size=[n,d_Y])
  index        = np.random.permutation(n)
  y_permute    = y[index, :]

  y_jx         = jnp.array(y_permute)
  x_jx         = jnp.array(x)
  w_jx         = jnp.array(w)

  return mu, nv, y_jx, x_jx, w_jx

def sample_circle(n):
  theta_span = np.linspace(0, 2*np.pi, n)
  x = np.array([np.cos(theta_span), np.sin(theta_span)]).T
  y = x
  a = np.ones(n)/n
  b = np.ones(n)/n
  x_jx = jnp.array(x)
  y_jx = jnp.array(y)
  a_jx = jnp.array(a)
  b_jx = jnp.array(b)

  return a_jx, b_jx, x_jx, y_jx


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces



class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))


    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

class RandRotation(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        rot_matrix = special_ortho_group.rvs(3)
        diag_matrix = np.diag(np.random.uniform(low=1, high=3, size=3))
        matrix = rot_matrix.dot(diag_matrix)

        rot_pointcloud = pointcloud @ matrix #matrix.dot(pointcloud.T).T
        return  rot_pointcloud, matrix

class RandomNoise(object):
    def __call__(self, pointcloud, noise_level):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0,noise_level, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

def room(n1, n2 ,n3, vector1, vector2):
    with open("ModelNet10/chair/train/chair_0001.off", 'r') as f:
        verts1, faces1 = read_off(f)

    with open("ModelNet10/desk/train/desk_0001.off", 'r') as f:
        verts2, faces2 = read_off(f)

    with open("ModelNet10/sofa/train/sofa_0001.off", 'r') as f:
        verts3, faces3 = read_off(f)

    point1 = PointSampler(n1)((verts1, faces1))
    point2 = PointSampler(n2)((verts2, faces2))
    point3 = PointSampler(n3)((verts3, faces3))

    point = np.concatenate([point1+vector1, point2,point3+vector2], axis=0)

    return point

def noisy_rot_point(n, point, noise_level, seed):
    np.random.seed(seed)
    norm_pointcloud        = Normalize()(point)
    rot_pointcloud, matrix = RandRotation()(norm_pointcloud)
    noisy_rot_pointcloud   = RandomNoise()(rot_pointcloud,noise_level )
    index                  = np.random.permutation(n)
    noisy_rot_pointcloud_permute = noisy_rot_pointcloud[index, :]
    return norm_pointcloud, noisy_rot_pointcloud_permute, matrix

