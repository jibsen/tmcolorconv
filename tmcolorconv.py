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
Convert tmTheme colors from Generic RGB to sRGB.

Many TextMate color themes were designed on Mac OS X, which used the Generic
RGB color profile by default (though apparently unmanaged applications get
sRGB on OS X 10.7 (2011) and later, but TextMate uses Generic RGB).

For portability and compatibility with other editors and operating systems,
it is better to use sRGB.

To complicate things, the default gamma value was changed from 1.8 to 2.2 in
OS X 10.6 (2009, http://support.apple.com/kb/ht3712), and TextMate 2 up to
a9290 applied alpha by blending with the previous color setting for the
element instead of being transparent against the background.

This script will convert the color values in a tmTheme file from Generic RGB
to sRGB, optionally applying simple alpha blending with the foreground and
background colors.

The blending assumes that the foreground and background keys come before any
other keys that contain alpha.

Note:
    Uses plistlib interface introduced in Python 3.4.
"""

import argparse
import collections
import math
import operator
import plistlib
import re


# Generated using mkconvmatrix.py
GenericRGBtosRGB = [[1.0252482, -0.0265428, 0.0012946],
                    [0.0193970, 0.9480316, 0.0325715],
                    [-0.0017702, -0.0014426, 1.0032129]]


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


def str_to_color(s):
    """Convert hex string to color value."""
    if len(s) == 3:
        s = ''.join(c + c for c in s)

    values = bytes.fromhex(s)

    # Scale from [0-255] to [0-1]
    return [c / 255.0 for c in values]


def color_to_str(C):
    """Convert color value to hex string."""
    # Scale from [0-1] to [0-255]
    V = [int(round(c * 255)) for c in C]

    # Clamp values to [0-255]
    for i in range(len(V)):
        V[i] = max(min(V[i], 255), 0)

    return '#{:02X}{:02X}{:02X}'.format(V[0], V[1], V[2])


def alpha_blend(color, base, alpha):
    """Blend color and base based on alpha."""
    return [c * alpha + b * (1 - alpha) for c, b in zip(color, base)]


def sRGB_compand(c):
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * math.pow(c, 1 / 2.4) - 0.055


def convert_color(Vin, gamma):
    """Convert color Vin from Generic RGB to sRGB."""
    # Linearize
    v = [math.pow(c, gamma) for c in Vin]

    v_srgb = mat_vec_mul(GenericRGBtosRGB, v)

    # sRGB companding
    Vout = list(map(sRGB_compand, v_srgb))

    return Vout


def convert_scheme(scheme, gamma, blend_alpha):
    """Convert colors in scheme from Generic RGB to sRGB.

    Args:
        scheme: tmTheme loaded through plistlib.
        gamma (float): Gamma value of colors.
        blend_alpha (bool): If True, colors with alpha are blended, otherwise
            the alpha value is copied.

    Returns:
        Converted scheme.
    """
    bg = [0, 0, 0]
    fg = [0, 0, 0]

    if gamma == 2.2:
        gamma = 563 / 256.0
    elif gamma == 1.8:
        gamma = 461 / 256.0

    for idx, entry in enumerate(scheme['settings']):
        for k, v in entry['settings'].items():
            # Match 6 digit hex color with optional alpha
            match = re.match('#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?', v)

            # Match 3 digit hex color
            if not match:
                match = re.match('#([0-9a-fA-F]{3})', v)

            if match:
                color = str_to_color(match.group(1))

                alpha_str = match.group(2)

                # Blend alpha if present
                if blend_alpha and alpha_str:
                    alpha = int(alpha_str, 16) / 255.0
                    alpha_str = None

                    if k in ('background', 'lineHighlight', 'selection'):
                        color = alpha_blend(color, bg, alpha)
                    else:
                        color = alpha_blend(color, fg, alpha)

                # Update fg and bg if in editor settings
                if idx == 0:
                    if k == 'foreground':
                        fg = color
                    elif k == 'background':
                        bg = color

                # Update hex color in scheme
                color_str = color_to_str(convert_color(color, gamma))
                color_str += alpha_str or ''
                scheme['settings'][idx]['settings'][k] = color_str

    return scheme


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Generic RGB to sRGB.')
    parser.add_argument('infile', type=argparse.FileType('rb'),
                        help='input tmTheme file')
    parser.add_argument('outfile', type=argparse.FileType('wb'),
                        help='output tmTheme file')
    parser.add_argument('-g', '--gamma', type=float, default=1.8,
                        help='input gamma (default 1.8)')
    parser.add_argument('-b', action='store_true',
                        help='blend alpha')
    args = parser.parse_args()

    scheme = plistlib.load(args.infile, dict_type=collections.OrderedDict)

    if scheme.get('colorSpaceName') == 'sRGB':
        print('Warning: colorSpaceName key is already sRGB')
    else:
        scheme['colorSpaceName'] = 'sRGB'

    convert_scheme(scheme, args.gamma, args.b)

    plistlib.dump(scheme, args.outfile, sort_keys=False)
