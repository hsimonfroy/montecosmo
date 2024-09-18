from argparse import ArgumentParser
from pmesh.pm import ParticleMesh
ap = ArgumentParser()
ap.add_argument("config")

from fastpm.core import Solver
from fastpm.core import leapfrog
from fastpm.core import autostages
from fastpm.background import PerturbationGrowth

from nbodykit.cosmology import Planck15
from nbodykit.cosmology import EHPower
from nbodykit.cosmology import Cosmology
from nbodykit.lab import FFTPower, FieldMesh
import numpy as np

from nbodykit.source.catalog.file import BigFileCatalog
from nbodykit.source.mesh import BigFileMesh


def main(args=None):
    path = "./fastpm/"
    part = BigFileCatalog(path+'fpm-1.0000', dataset='1/', header='Header')
    np.save(path+"part.npy", part)
    mesh = BigFileMesh(path+'fpm-1.0000/', 'Header')
    np.save(path+"mesh.npy", mesh)



if __name__ == '__main__':
    main()