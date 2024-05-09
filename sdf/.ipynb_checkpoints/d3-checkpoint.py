import functools
import numpy as np
import operator
import torch
from . import dn, d2, ease, mesh

# Constants

ORIGIN = torch.Tensor([0,0,0])
# ORIGIN = np.array([0,0,0])
# SDF Class

_ops = {}

class SDF3:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        return self.f(p)#.reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        self._k = k
        return self
    def generate(self, *args, **kwargs):
        return mesh.generate(self, *args, **kwargs)
    def save(self, path, *args, **kwargs):
        return mesh.save(path, self, *args, **kwargs)
    def show_slice(self, *args, **kwargs):
        return mesh.show_slice(self, *args, **kwargs)

def sdf3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    return wrapper

def op3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op32(f):
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

# Helpers

def _length2(a):
    return torch.norm(a, p=2, dim=-1)

def _length(a):
    return torch.norm(a, p=None, dim=-1)

def _normalize(a):
    return a / np.linalg.norm(a)

def _dot(a, b):
    return np.sum(a * b, axis=1)

def _vec(arrs):
    return torch.cat(arrs, dim=-1)

def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

@sdf3
def sphere(size=1, center=ORIGIN):
    def f(p):
        ctx = p.device
        zero_tensor = torch.tensor(0).to(ctx)
        center_ = center.to(ctx)
        size_tensor = size.to(ctx)[:,:,5:6].unsqueeze(-1) #[bs x 1 x 3]
        return _length(p - center_) - size_tensor[:,:,0]
    return f

@sdf3
def cylinder(size=1):
    def f(p):
        ctx = p.device
        zero_tensor = torch.tensor(0).to(ctx)
        ra = size[:,:,3:4]
        h = size[:,:,4:5]
        d = _vec([
            _length(p[:,:,[0,2]]).unsqueeze(-1) - ra,
            torch.abs(p[:,:,1:2]) - h / 2])
        return (
            torch.minimum(torch.maximum(d[:,:,0], d[:,:,1]), zero_tensor) +
            _length(torch.maximum(d, zero_tensor)))
    return f

@sdf3
def ellipsoid(size, center=ORIGIN):
    def f(p):
        ctx = p.device
        zero_tensor = torch.tensor(0).to(ctx)
        center_ = center.to(ctx)
        size_tensor = size.to(ctx)[:,:,3].unsqueeze(-1) + 1e-6

        k0 = _length((p - center_) * size_tensor)
        k1 = _length((p - center_) * (size_tensor * size_tensor))
        return k0 * (k0 - 1) / (k1+1e-6)
    return f

@sdf3
def box(size=1, center=ORIGIN):
    def f(p):
        ctx = p.device
        zero_tensor = torch.tensor(0).to(ctx)
        center_ = center.to(ctx)
        size_tensor = size.to(ctx)[:,:,:3]
        q = torch.abs(p - center_) - size_tensor / 2
        
        return _length(torch.maximum(q, zero_tensor)) + torch.minimum(torch.max(q, dim=-1)[0], zero_tensor)
    return f

# Positioning

@op3
def translate(other, offset):
    def f(p):
        return other(p - offset)
    return f

@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))
    def f(p):
        return other(p / s) * m
    return f

@op3
def rotate(other, matrix):
    def f(p):
        return other(torch.einsum('abd,abde->abe',p,matrix))
        # return other(p @ matrix)
    return f

