"""QM9/GDB9-14 dataset.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
Matthias Rupp, 2020

134k small organic molecules, in their ground states, with energetic, electronic
and thermodynamic properties. 133,885 small organic molecules with up to 9 C, O, N, F atoms,
saturated with H. Geometries, harmonic frequencies, dipole moments, polarizabilities,
energies, enthalpies, and free energies of atomization at the DFT/B3LYP/6-31G(2df,p)
level of theory. For a subset of 6,095 constitutional isomers of C7H10O2, energies,
enthalpies, and free energies of atomization are provided at the G4MP2 level of theory.

See class Qm9RamakrishnanEtAl2014Dataset for details.
"""


import bz2
import os.path
import zipfile

import numpy as np
import pandas as pd

from smlb import element_data, TabularDataFromPandas, params


# todo: support automatic download of dataset from qmml.org

# todo: add isomers as own dataset


class Qm9RamakrishnanEtAl2014Dataset(TabularDataFromPandas):
    r"""134k small organic molecules with computed geometries and ground-state properties.

    Benchmarking dataset of 134k small organic molecules simulated at DFT/B3LYP/6-31G(2df,p) level
    of theory. Ground-state geometry and energetic, electronic and thermodynamic properties.

    133,885 small organic molecules with up to 9 C, O, N, F atoms, saturated with H.
    Provides geometries, harmonic frequencies, dipole moments, polarizabilities, energies,
    enthalpies, and free energies of atomization.

    3,054 molecules failed a consistency check where the Corina generated Cartesian coordinates
    and the B3LYP/6-31G(2df,p) equilibrium geometry lead to different SMILES strings.
    These molecules can be excluded.  # todo

    For 19 molecules there were convergence problems. These can be excluded.  # todo

    This dataset requires access to the underlying data file (85 MB),
    which can be downloaded from https://qmml.org (last accessed 2020-04-06).
    It expects the file names and file formats as they are in that data file.

    Reference:
    Raghunathan Ramakrishnan, Pavlo Dral, Matthias Rupp, O. Anatole von Lilienfeld:
    Quantum Chemistry Structures and Properties of 134 kilo Molecules,
    Scientific Data 1: 140022, 2014. DOI 10.1038/sdata.2014.22
    """

    def __init__(
        self,
        source: str,
        exclude_uncharacterized: bool = False,
        exclude_unconverged: bool = False,
        **kwargs,
    ):
        """Load dataset.

        See TabularDataFromPandas for parameters to control pre-processing on loading,
        such as joining, filtering, as well as sample and label transformations.

        Parameters:
            source: path to underlying data file (see class docstring)
            exclude_uncharacterized: exclude molecules listed in file 'uncharacterized.txt'
            exclude_unconverged: exclude molecules listed as hard to converge in file 'readme.txt'

        The files 'uncharacterized.txt' and 'readme.txt' are part of the original dataset.
        The indices of these uncharacterized and unconverged molecules are fixed; in particular,
        they are not loaded dynamically from the dataset.

        Samples:
            index: unique integer
            atomic_number: k-vector of atomic numbers (proton numbers)
            coordinates: k x 3 array of k 3d points (x,y,z)
            mulliken_charges: k-vector of Mulliken partial charges
            frequencies: frequencies (either 3k-5 or 3k-6)
            smiles_gdb9: SMILES encoding of the original GDB9 molecular structure graph
            smiles_relaxed: SMILES encoding of the relaxed-geometry molecular structure graph
            inchi_gdb9: InChI encoding of the original GDB9 molecular structure graph
            inchi_relaxed: InChI encoding of the relaxed-geometry molecular structure graph

        Labels:
             0  A       GHz          Rotational constant A
             1  B       GHz          Rotational constant B
             2  C       GHz          Rotational constant C
             3  mu      Debye        Dipole moment
             4  alpha   Bohr^3       Isotropic polarizability
             5  homo    Hartree      Energy of Highest occupied molecular orbital (HOMO)
             6  lumo    Hartree      Energy of Lowest occupied molecular orbital (LUMO)
             7  gap     Hartree      Gap, difference between LUMO and HOMO
             8  r2      Bohr^2       Electronic spatial extent
             9  zpve    Hartree      Zero point vibrational energy
            10  U0      Hartree      Internal energy at 0 K
            11  U       Hartree      Internal energy at 298.15 K
            12  H       Hartree      Enthalpy at 298.15 K
            13  G       Hartree      Free energy at 298.15 K
            14  Cv      cal/(mol K)  Heat capacity at 298.15 K

        Raises:
            InvalidParameterError: on invalid parameter values
        """

        # parameter validation
        source = params.string(source)  # todo: params.filename
        exclude_uncharacterized = params.boolean(exclude_uncharacterized)
        exclude_unconverged = params.boolean(exclude_unconverged)

        # load raw data
        # bunzip2 takes about 7s for this 85 MB file
        # therefore, support both reading the unpacked file or the packed ones
        if source[-4:] == ".xyz":  # unpacked
            with open(source, "tr") as f:
                raw = f.read()
        elif source[-8:] == ".xyz.bz2":  # bz2-packed
            with open(source, "br") as f:
                raw = bz2.decompress(f.read()).decode(encoding="ascii")
        elif source[-4:] == ".zip":  # bz2-packed within zip archive
            with zipfile.ZipFile(source) as zf:
                with zf.open("dsgdb9nsd.xyz.bz2") as f:  # filename as in downloaded dataset
                    raw = bz2.decompress(f.read()).decode(encoding="ascii")

        # parse data
        propnames = [
            "A",
            "B",
            "C",
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "U0",
            "U",
            "H",
            "G",
            "Cv",
        ]

        def parse(mol: str):
            lines = mol.split("\n")
            result = {}

            na = int(lines[0])  # number of atoms

            props = lines[1].split()  # gdb identifier, molecule's index, and properties 1-15
            assert props[0] == "gdb", "internal error: wrong file format parsing QM9 molecule"
            result["index"] = int(props[1])
            assert len(propnames) == len(props[2:]), "internal error parsing QM9 molecule"
            for key, value in zip(propnames, props[2:],):
                result[key] = float(value)

            atomblock = np.array([line.split() for line in lines[2 : na + 2]])  # array of strings

            result["atomic_number"] = [element_data(an, "Z") for an in atomblock[:, 0]]
            result["coordinates"] = np.asfarray(atomblock[:, 1:4])
            result["mulliken_charges"] = np.asfarray(atomblock[:, 4])

            result["frequencies"] = np.asfarray(lines[na + 2].split())
            result["smiles_gdb9"], result["smiles_relaxed"] = lines[na + 3].split()
            result["inchi_gdb9"], result["inchi_relaxed"] = lines[na + 4].split()

            return result

        # alternative via qmmlpack:
        # qmml.import_extxyz(raw, additional_properties=True)

        parsed = [parse(entry) for entry in raw.split("\n\n")]
        data = pd.DataFrame(parsed)

        # drop molecule subsets if requested
        if exclude_uncharacterized:
            filename = os.path.join(os.path.dirname(__file__), "uncharacterized.txt")
            with open(filename, "rt") as f:
                excluded = f.read().split("\n")
                while not excluded[0].startswith("  "):  # drop header lines
                    del excluded[0]
                while not excluded[-1].startswith("  "):  # drop footer lines
                    del excluded[-1]
                excluded = [int(line.split()[0]) for line in excluded]
            data = data[~data["index"].isin(excluded)]

        if exclude_unconverged:
            excluded = [
                21725,
                87037,
                59827,
                117523,
                128113,
                129053,
                129152,
                129158,
                130535,
                6620,
                59818,
            ]
            data = data[~data["index"].isin(excluded)]

        super().__init__(data=data, labels=propnames, **kwargs)
