from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from netCDF4 import Dataset as NetCDFFile
from jigsawpy import jigsaw_msh_t, loadmsh, pwrbal2

import argparse


def jigsaw_to_netcdf(msh_filename, output_name, on_sphere, sphere_radius=None):
    """
    Converts mesh data defined in JIGSAW format to NetCDF

    Parameters
    ----------
    msh_filename : str
        A JIGSAW mesh file name
    output_name: str
        The name of the output file
    on_sphere : bool
        Whether the mesh is spherical or planar
    sphere_radius : float, optional
        The radius of the sphere in meters.  If ``on_sphere=True`` this argument
        is required, otherwise it is ignored.
    """
    # Authors: Phillip J. Wolfram, Matthew Hoffman, Xylar Asay-Davis
    #          and Darren Engwirda

    grid = NetCDFFile(output_name, 'w', format='NETCDF3_CLASSIC')

    # Get dimensions
    # Get nCells
    msh = jigsaw_msh_t()
    loadmsh(msh_filename, msh)
    nCells = msh.point.shape[0]

    # Get vertexDegree and nVertices
    vertexDegree = 3  # always triangles with JIGSAW output
    nVertices = msh.tria3.shape[0]

    if vertexDegree != 3:
        ValueError('This script can only compute vertices with triangular '
                   'dual meshes currently.')

    grid.createDimension('nCells', nCells)
    grid.createDimension('nVertices', nVertices)
    grid.createDimension('vertexDegree', vertexDegree)

    # Create cell variables and sphere_radius
    if msh.vert3.size > 0:
        xCell_full = msh.vert3['coord'][:, 0]
        yCell_full = msh.vert3['coord'][:, 1]
        zCell_full = msh.vert3['coord'][:, 2]
    else:
        xCell_full = msh.vert2['coord'][:, 0]
        yCell_full = msh.vert2['coord'][:, 1]
        zCell_full = np.zeros(nCells, dtype=float)

    if msh.power.size > 0:
        wCell_full = msh.power[:]
    else:
        wCell_full = np.zeros(nCells, dtype=float)

    for cells in [xCell_full, yCell_full, zCell_full, wCell_full]:
        assert cells.shape[0] == nCells, \
            'Number of anticipated nodes is not correct!'

    if on_sphere:
        grid.on_a_sphere = "YES"
        grid.sphere_radius = sphere_radius
        # convert from km to meters
        xCell_full *= 1e3
        yCell_full *= 1e3
        zCell_full *= 1e3
        wCell_full *= 1e6   # w = r^2
    else:
        grid.on_a_sphere = "NO"
        grid.sphere_radius = 0.0

    # Create cellsOnVertex
    cellsOnVertex_full = msh.tria3['index'] + 1
    assert cellsOnVertex_full.shape == (nVertices, vertexDegree), \
        'cellsOnVertex_full is not the right shape!'

    # Create vertex variables
    ball = pwrbal2(
        np.hstack((xCell_full, yCell_full, zCell_full)),
        wCell_full, cellsOnVertex_full - 1)

    if on_sphere:
        ball = to_sphere(sphere_radius, ball)

    xVertex_full = ball[:, 0]
    yVertex_full = ball[:, 1]
    zVertex_full = ball[:, 2]

    meshDensity_full = grid.createVariable(
        'meshDensity', 'f8', ('nCells',))

    for iCell in np.arange(0, nCells):
        meshDensity_full[iCell] = 1.0

    del meshDensity_full

    var = grid.createVariable('xCell', 'f8', ('nCells',))
    var[:] = xCell_full
    var = grid.createVariable('yCell', 'f8', ('nCells',))
    var[:] = yCell_full
    var = grid.createVariable('zCell', 'f8', ('nCells',))
    var[:] = zCell_full
    var = grid.createVariable('wCell', 'f8', ('nCells',))
    var[:] = wCell_full
    var = grid.createVariable('featureTagCell', 'i4', ('nCells',))
    var[:] = msh.point['IDtag']
    var = grid.createVariable('xVertex', 'f8', ('nVertices',))
    var[:] = xVertex_full
    var = grid.createVariable('yVertex', 'f8', ('nVertices',))
    var[:] = yVertex_full
    var = grid.createVariable('zVertex', 'f8', ('nVertices',))
    var[:] = zVertex_full
    var = grid.createVariable('featureTagVertex', 'i4', ('nVertices',))
    var[:] = msh.tria3['IDtag']
    var = grid.createVariable(
        'cellsOnVertex', 'i4', ('nVertices', 'vertexDegree',))
    var[:] = cellsOnVertex_full

    grid.sync()
    grid.close()


def to_sphere(radii, E3):
    """
    TO-SPHERE: project (geocentric) a set of points in R^3
    onto a spheroidal surface.

    """
    # Authors: Darren Engwirda
    
    if (radii.size == 1):
        radii = radii * np.ones(3, dtype=float)

    PP = .5 * E3

    ax = PP[:, 0] ** 1 / radii[0] ** 1
    ay = PP[:, 1] ** 1 / radii[1] ** 1
    az = PP[:, 2] ** 1 / radii[2] ** 1

    aa = ax ** 2 + ay ** 2 + az ** 2

    bx = PP[:, 0] ** 2 / radii[0] ** 2
    by = PP[:, 1] ** 2 / radii[1] ** 2
    bz = PP[:, 2] ** 2 / radii[2] ** 2

    bb = bx * 2. + by * 2. + bz * 2.

    cx = PP[:, 0] ** 1 / radii[0] ** 1
    cy = PP[:, 1] ** 1 / radii[1] ** 1
    cz = PP[:, 2] ** 1 / radii[2] ** 1

    cc = cx ** 2 + cy ** 2 + cz ** 2
    cc = cc - 1.0

    ts = bb * bb - 4. * aa * cc

    ok = ts >= .0

    AA = aa[ok]; BB = bb[ok]; CC = cc[ok]; TS = ts[ok]

    t1 = (-BB + np.sqrt(TS)) / AA / 2.0
    t2 = (-BB - np.sqrt(TS)) / AA / 2.0

    tt = np.maximum(t1, t2)
    
    P3 = np.zeros(E3.shape, dtype=float)
    P3[ok, 0] = (1.0 + tt) * PP[ok, 0]
    P3[ok, 1] = (1.0 + tt) * PP[ok, 1]
    P3[ok, 2] = (1.0 + tt) * PP[ok, 2]

    return P3


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-m",
        "--msh",
        dest="msh",
        required=True,
        help="input .msh file generated by JIGSAW.",
        metavar="FILE")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="grid.nc",
        help="output file name.",
        metavar="FILE")
    parser.add_argument(
        "-s",
        "--spherical",
        dest="spherical",
        action="store_true",
        default=False,
        help="Determines if the input/output should be spherical or not.")
    parser.add_argument(
        "-r",
        "--sphere_radius",
        dest="sphere_radius",
        type=float,
        help="The radius of the sphere in meters")

    args = parser.parse_args()

    jigsaw_to_netcdf(args.msh, args.output, args.spherical, args.sphere_radius)
