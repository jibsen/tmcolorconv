#!/usr/bin/env python3

# Copyright (c) 2014 Joergen Ibsen
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Functionality for making RGB working space conversion matrices.

This code was written while working on a post about how color management
affects color themes for text editors.

Some useful resources:

http://www.brucelindbloom.com/
http://www.babelcolor.com/download/A%20review%20of%20RGB%20color%20spaces.pdf
http://www.marcelpatek.com/color.html
http://ninedegreesbelow.com/photography/articles.html
http://www.ryanjuckett.com/programming/rgb-color-space-conversion/

Warning:
    This is a playground, not a product.
"""

import collections
import math
import operator


class Chromaticity(collections.namedtuple('Chromaticity', 'x y')):
    __slots__ = ()

    @classmethod
    def from_XYZ(cls, X, Y, Z):
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        return cls(x, y)


class WhitePoint(collections.namedtuple('WhitePoint', 'X Y Z')):
    __slots__ = ()

    @classmethod
    def from_xy(cls, x, y):
        X = x / y
        Y = 1.0
        Z = (1.0 - x - y) / y
        return cls(X, Y, Z)


ColorSpace = collections.namedtuple('ColorSpace', 'r g b wp tf')


class SimpleCompander:
    def __init__(self, gamma):
        self._gamma = gamma

    def compand(self, c):
        return math.pow(max(c, 0.0), 1.0 / self._gamma)

    def linearize(self, c):
        return math.pow(max(c, 0.0), self._gamma)


class sRGBCompander:
    def compand(self, c):
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * math.pow(c, 1 / 2.4) - 0.055

    def linearize(self, c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return math.pow((c + 0.055) / 1.055, 2.4)


# White points
# https://en.wikipedia.org/wiki/Standard_illuminant
# ASTM E308-01
# ICC values from http://www.color.org/ICC_Minor_Revision_for_Web.pdf

D50 = WhitePoint(0.96422, 1.0, 0.82521)
D50ICC = WhitePoint(0.9642, 1.0, 0.8249)

D65 = WhitePoint(0.95047, 1.0, 1.08883)
D65ICC = WhitePoint(0.9505, 1.0, 1.0890)

# Color component transfer functions (companding)

GAMMA18 = 461 / 256.0
GAMMA22 = 563 / 256.0

tf_g18 = SimpleCompander(GAMMA18)
tf_g22 = SimpleCompander(GAMMA22)
tf_sRGB = sRGBCompander()

# Primaries

prim_AdobeRGB = (Chromaticity(0.64, 0.33),
                 Chromaticity(0.21, 0.71),
                 Chromaticity(0.15, 0.06))

prim_HDTV_709 = (Chromaticity(0.64, 0.33),
                 Chromaticity(0.3, 0.6),
                 Chromaticity(0.15, 0.06))

prim_P22_EBU = (Chromaticity(0.63, 0.34),
                Chromaticity(0.295, 0.605),
                Chromaticity(0.15, 0.075))

# Generic RGB uses this slightly different version of P22
prim_P22_alt = (Chromaticity(0.63, 0.34),
                Chromaticity(0.295, 0.605),
                Chromaticity(0.155, 0.077))

prim_Trinitron = (Chromaticity(0.625, 0.34),
                  Chromaticity(0.28, 0.595),
                  Chromaticity(0.155, 0.07))

# RGB color spaces

# Using white point D65 gives the conversion matrices of Lindbloom/Pascale,
# whereas using 0.3127, 0.3290 as in the AdobeRGB spec gives theirs.
# https://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
AdobeRGB = ColorSpace(*prim_AdobeRGB,
                      wp=D65,
                      # wp=WhitePoint.from_xy(0.3127, 0.3290),
                      tf=tf_g22)

AppleRGB = ColorSpace(*prim_Trinitron,
                      wp=D65,
                      tf=tf_g18)

# https://developer.apple.com/library/mac/qa/qa1430/_index.html
GenericRGB = ColorSpace(*prim_P22_alt,
                        wp=D65,
                        tf=tf_g18)

# https://en.wikipedia.org/wiki/SRGB
# The sRGB spec white point is 0.3127, 0.3290, we use D65 to match Lindbloom.
sRGB = ColorSpace(*prim_HDTV_709,
                  wp=D65,
                  # wp=WhitePoint.from_xy(0.3127, 0.3290),
                  tf=tf_sRGB)


# Matrix computations (yes, numpy is a lot faster)
def mat_c_mul(M, c):
    """Multiply matrix by constant."""
    return [[a * c for a in r] for r in M]


def mat_vec_mul(M, v):
    """Multiply matrix by vector."""
    return [sum(map(operator.mul, r, v)) for r in M]


def mat_mat_mul(M1, M2):
    """Multiply matrix by matrix."""
    M2T = list(zip(*M2))
    return [mat_vec_mul(M2T, r) for r in M1]


def mat3_inv(M):
    """Compute inverse of 3x3 matrix."""
    adjM = [[None, None, None] for _ in range(3)]

    # Compute the adjugate of M
    # Note: Indexing handles transposing and sign change (only works for 3x3)
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            adjM[j][i] = M[i1][j1] * M[i2][j2] - M[i1][j2] * M[i2][j1]

    det = sum(M[0][i] * adjM[i][0] for i in range(3))

    return mat_c_mul(adjM, 1 / det)


def make_cs_to_XYZ_matrix(cs):
    """Compute matrix to convert from color space cs to XYZ.

    Based on Foley et al.

    Args:
        cs (ColorSpace): Source color space.

    Returns:
        Conversion matrix.
    """
    zr = 1.0 - cs.r.x - cs.r.y
    zg = 1.0 - cs.g.x - cs.g.y
    zb = 1.0 - cs.b.x - cs.b.y

    C = [[cs.r.x, cs.g.x, cs.b.x],
         [cs.r.y, cs.g.y, cs.b.y],
         [zr, zg, zb]]

    Cinv = mat3_inv(C)

    T = mat_vec_mul(Cinv, cs.wp)

    M = [list(map(operator.mul, r, T)) for r in C]

    return M


def make_bfd_matrix(ws, wd):
    """Compute Bradford chromatic adaption transform matrix.

    Args:
        ws (WhitePoint): Source white point (XYZ).
        wd (WhitePoint): Destination white point (XYZ).

    Returns:
        Chromatic adapation matrix.
    """
    # Bradford cone response matrix
    Bradford_crm = [[0.8951, 0.2664, -0.1614],
                    [-0.7502, 1.7135, 0.0367],
                    [0.0389, -0.0685, 1.0296]]

    Cs = mat_vec_mul(Bradford_crm, ws)
    Cd = mat_vec_mul(Bradford_crm, wd)

    T = [[Cd[0] / Cs[0], 0.0, 0.0],
         [0.0, Cd[1] / Cs[1], 0.0],
         [0.0, 0.0, Cd[2] / Cs[2]]]

    return mat_mat_mul(mat3_inv(Bradford_crm), mat_mat_mul(T, Bradford_crm))


def point_in_triangle(p, a, b, c):
    """Check if p is inside or on edge of triangle a, b, c."""
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])

    u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
    v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det

    return u >= 0.0 and v >= 0.0 and (u + v) <= 1.0


def barycentric_clamp(p, a, b, c):
    """Clamp p to triangle a, b, c using barycentric coordinate system.

    Convert p into area coordinates (barycentric coordinate system) of
    triangle with corners a, b, c, and if p is outside, move it to a
    position on the edge "nearby".

    This process is not based on any color theory, but is an easy and fast
    way to clamp coordinates.
    """
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])

    u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
    v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
    w = 1.0 - u - v

    if u < 0.0:
        u = 0.0
        v /= v + w
        w = 1.0 - v

    if v < 0.0:
        v = 0.0
        u /= u + w
        w = 1.0 - u

    if w < 0.0:
        w = 0.0
        u /= u + v
        v = 1.0 - u

    return [u * a[0] + v * b[0] + w * c[0], u * a[1] + v * b[1] + w * c[1]]


def point_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def closest_point_on_line(p, a, b):
    """Find closest point to p on line a, b."""
    ablen = point_distance(a, b)

    # Unit vector along a, b
    abux = (b[0] - a[0]) / ablen
    abuy = (b[1] - a[1]) / ablen

    # Length of projection of a, p onto a, b
    t = (p[0] - a[0]) * abux + (p[1] - a[1]) * abuy

    if t < 0.0:
        return a
    elif t > ablen:
        return b
    else:
        return [a[0] + abux * t, a[1] + abuy * t]


def closest_point_on_triangle(p, a, b, c):
    """Find closest point to p on edge of triangle a, b, c."""
    p1 = closest_point_on_line(p, a, b)
    p2 = closest_point_on_line(p, b, c)
    p3 = closest_point_on_line(p, c, a)

    d1 = point_distance(p, p1)
    d2 = point_distance(p, p2)
    d3 = point_distance(p, p3)

    result = p1
    dresult = d1

    if d2 < dresult:
        result = p2
        dresult = d2

    if d3 < dresult:
        result = p3
        dresult = d3

    return result


class RGBConverter:
    def __init__(self, src_cs, dst_cs):
        self.src = src_cs
        self.dst = dst_cs
        self._src_to_XYZ = make_cs_to_XYZ_matrix(src_cs)
        self._XYZ_to_dst = mat3_inv(make_cs_to_XYZ_matrix(dst_cs))
        self._cat = None
        if src_cs.wp != dst_cs.wp:
            self._cat = make_bfd_matrix(src_cs.wp, dst_cs.wp)

    def convert(self, C):
        """Convert color C from src to dst colorspace."""
        C = list(map(self.src.tf.linearize, C))

        C = mat_vec_mul(self._src_to_XYZ, C)

        if self._cat is not None:
            C = mat_vec_mul(self._cat, C)

        C = mat_vec_mul(self._XYZ_to_dst, C)

        C = list(map(self.dst.tf.compand, C))

        # Clamp values to [0-1]
        C = [min(max(c, 0.0), 1.0) for c in C]

        return C

    def convert_hex(self, s):
        C = [c / 255.0 for c in bytes.fromhex(s)]

        C = self.convert(C)

        C = [int(round(c * 255)) for c in C]

        return '#{:02X}{:02X}{:02X}'.format(*C)


if __name__ == '__main__':
    D50toD65 = make_bfd_matrix(D50, D65)
    print('D50toD65:', D50toD65)
    print('white:', mat_vec_mul(D50toD65, D50))

    M1 = make_cs_to_XYZ_matrix(GenericRGB)
    M2 = make_cs_to_XYZ_matrix(sRGB)
    M = mat_mat_mul(mat3_inv(M2), M1)
    print('GenericRGB to sRGB:', M)
    print('white:', mat_vec_mul(M, (1.0, 1.0, 1.0)))
    print('black:', mat_vec_mul(M, (0.0, 0.0, 0.0)))
    print('75 gray:', mat_vec_mul(M, (0.75, 0.75, 0.75)))
    print('M1 row sums :', [sum(r) for r in M1])
    print('M2 row sums :', [sum(r) for r in M2])
    print('M row sums :', [sum(r) for r in M])

    conv = RGBConverter(GenericRGB, sRGB)

    print(conv.convert_hex('272822'))
    print(conv.convert_hex('F92672'))
    print(conv.convert_hex('FF0000'))
    print(conv.convert_hex('00FF00'))
    print(conv.convert_hex('0000FF'))

    # To get the primaries from a color profile, we must convert the rXYZ,
    # gXYZ, bXYZ entries to chromaticities. This example converts the values
    # from a sRGB profile. The values are stored relative to whitepoint D50,
    # so we must adapt them.
    rXYZ = [0.43606567, 0.22248840, 0.01391602]
    gXYZ = [0.38514709, 0.71687317, 0.09707642]
    bXYZ = [0.14306641, 0.06060791, 0.71409607]
    r = mat_vec_mul(D50toD65, rXYZ)
    g = mat_vec_mul(D50toD65, gXYZ)
    b = mat_vec_mul(D50toD65, bXYZ)
    print('sRGB color profile primaries converted to chromaticities:')
    print('R 0.64 0.33:', Chromaticity.from_XYZ(*r))
    print('G 0.30 0.60:', Chromaticity.from_XYZ(*g))
    print('B 0.15 0.06:', Chromaticity.from_XYZ(*b))
