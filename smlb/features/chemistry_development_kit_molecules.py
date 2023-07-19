"""Chemistry Development Kit (CDK) molecular features.

Scientific Machine Learning Benchmark
A benchmark of regression models in chem- and materials informatics.
2019-2020, Citrine Informatics.

See docstring of ChemistryDevelopmentKitMoleculeFeatures for details.
"""

# There are also dedicated wrappers for CDK.
# However, these proved less suitable than generic bridging to Java.
# * Cinfony Python wrapper. Also wraps OpenBabel and RDKit.
#   Outdated (Python 2); could be salvaged, but support unclear.
# * Sebotic's wrapper at https://github.com/sebotic/cdk_pywrapper
#   Unsuitable quality, maintenance doubtful. Uses py4j internally.

import os
from typing import Any, Callable, Optional, Sequence

import numpy as np

import py4j.java_gateway

from smlb import (
    BenchmarkError,
    InvalidParameterError,
    Data,
    DataTransformationFailureMode,
    Features,
    params,
    TabularData,
)
from smlb.core.java import JavaGateway

# todo: get column types and names from CDK descriptors, set accordingly (requires named column support)
# todo: detect changed path to CDK jar on repeated calls and re-initialize JVM if necessary


class CdkJavaGateway(JavaGateway):
    """Provide gateway to CDK Java functionality."""

    _cdk_jar_path_auto = None  # auto-detected path to CDK .jar file

    def __init__(self, cdk_jar_path: Optional[str] = None):
        """Initialize CDK Java gateway.

        See base class JavaGateway for details.

        This class provides CDK-specific functionality,
        namely the path to the CDK .jar file.

        Parameters:
            cdk_jar_path: local filesystem path to the CDK jar, e.g.,
                '/file/path/cdk.jar'. If not specified, smlb tries to
                find the CDK jar.

        Raises:
            InvalidParameterError if the CDK .jar file can not be found.
        """

        # todo: optional_
        # cdk_jar_path = params.optional_(cdk_jar_path, params.string)  todo: valid path
        cdk_jar_path = params.any_(cdk_jar_path, params.string, params.none)

        # finding CDK .jar file logic
        if cdk_jar_path is None:
            if self._cdk_jar_path_auto is not None:
                # already detected, use stored path
                cdk_jar_path = self._cdk_jar_path_auto
            else:
                # attempt to find CDK .jar file
                # todo: find correct path for installed versions

                cdk_jar_path = os.path.join(os.path.dirname(__file__), "../build/cdk.jar")

        if not os.access(cdk_jar_path, os.R_OK):
            raise InvalidParameterError(
                "Valid path to .jar file",
                cdk_jar_path,
                explanation=f"Jar file {cdk_jar_path} does not exist or is not readable.",
            )

        super().__init__(cdk_jar_path)


class ChemistryDevelopmentKitMoleculeFeatures(Features):
    """Chemistry Development Kit molecular features.

    Based on the Chemistry Development Kit (CDK) package.

    CDK is a library, written in Java, that provides diverse chem(o)informatics and
    bioinformatics functionality. This class uses CDK's QSAR (quantitative structure-
    activity relationship) descriptor calculation (see Steinbeck et al. 2006).
    For further information on CDK, see links below.

    Computing descriptors via this class requires knowledge of the bonding in a molecule
    in the form of SMILES (Simplified Molecular-Input Line-Entry System) strings.

    References:

    Egon L. Willighagen, John W. Mayfield, Jonathan Alvarsson, Arvid Berg, Lars Carlsson,
    Nina Jeliazkova, Stefan Kuhn, Tomáš Pluskal, Miquel Rojas-Chertó, Ola Spjuth,
    Gilleain Torrance, Chris T. Evelo, Rajarshi Guha, Christoph Steinbeck:
    The Chemistry Development Kit (CDK) v2.0: atom typing,  depiction,  molecular
    formulas, and substructure searching, Journal of Cheminformatics 9(1): 33,
    2017. DOI 10.1186/s13321-017-0220-4

    Christoph Steinbeck, Christian Hoppe, Stefan Kuhn, Matteo Floris, Rajarshi Guha,
    Egon Willighagen: Recent Developments of the Chemistry Development Kit (CDK)---
    An Open-Source Java Library for Chemo- and Bioinformatics, Current Pharmaceutical
    Design 12(17): 2111-2120, 2006. DOI 10.2174/138161206777585274

    Christoph Steinbeck, Yongquan Han, Stefan Kuhn, Oliver Horlacher, Edgar Luttmann,
    Egon Willighagen: The Chemistry Development Kit (CDK): An Open-Source {J}ava
    Library for Chemo- and Bioinformatics, Journal of Chemical Information and Computer
    Sciences 43(2): 57-70, 2003. DOI 10.1021/ci025584y

    Code and documentation:
        https://cdk.github.io/
        https://github.com/cdk/cdk
    """

    # available descriptors
    DESCRIPTORS = {
        name: ("org.openscience.cdk.qsar.descriptors.molecular." + cdk_class, arity)
        for name, cdk_class, arity, _ in (
            (
                "acidic_group_count",
                "AcidicGroupCountDescriptor",
                1,
                "number of acidic groups as per SMARTS in reference. "
                "counts specific subgroups. "
                "https://doi.org/10.1002/qsar.200510009",
            ),
            (
                "alogp",
                "ALOGPDescriptor",
                3,
                "returns Ghose-Crippen LogKow, ALogP2, molar refractivity. "
                "H, C, O, N, S, halogens. "
                "assumes that aromaticity has been detected and hydrogens are explicit. "
                "https://doi.org/10.1002/jcc.540070419 "
                "https://doi.org/10.1021/ci00053a005",
            ),
            (
                "apol",  # atomic polarizability
                "APolDescriptor",
                1,
                "sum of the atomic polarizabilities (including implicit hydrogens). "
                "Polarizabilities are taken from http://www.sunysccc.edu/academic/mst/ptable/p-table2.htm.",
            ),
            (
                "aromatic_atoms_count",
                "AromaticAtomsCountDescriptor",
                1,
                "number of aromatic atoms. " "has a checkAromaticity: bool parameter. ",
            ),
            (
                "aromatic_bonds_count",
                "AromaticBondsCountDescriptor",
                1,
                "number of aromatic bonds. " "has a checkAromaticity: bool parameter",
            ),
            (
                "atom_count",
                "AtomCountDescriptor",
                1,
                "number of atoms. "
                "has an elementName: str parameter. "
                "could be varied to count by atom type. ",
            ),
            (
                "auto_correlation_charge",
                "AutocorrelationDescriptorCharge",
                5,
                "autocorrelation of topological structure (ATS), with weights = charges. "
                "difference to auto_correlation_polarizability not documented. "
                "G. Moreau G., P. Broto: The autocorrelation of a topological structure: "
                "A new molecular descriptor, Nouveau Journal de Chimie 359-360, 1980",
            ),
            (
                "auto_correlation_mass",
                "AutocorrelationDescriptorMass",
                5,
                "autocorrelation of topological structure (ATS), with weights = scaled atomic mass. "
                "G. Moreau G., P. Broto: The autocorrelation of a topological structure: "
                "A new molecular descriptor, Nouveau Journal de Chimie 359-360, 1980",
            ),
            (
                "auto_correlation_polarizability",
                "AutocorrelationDescriptorPolarizability",
                5,
                "autocorrelation of topological structure (ATS), with weights = charges. "
                "G. Moreau G., P. Broto: The autocorrelation of a topological structure: "
                "A new molecular descriptor, Nouveau Journal de Chimie 359-360, 1980",
            ),
            (
                "basic_group_count",
                "BasicGroupCountDescriptor",
                1,
                "groups as per SMARTS in reference. " "https://doi.org/10.1002/qsar.200510009",
            ),
            (
                "bcut",
                "BCUTDescriptor",
                6,
                "eigenvalue-based descriptor related to chemical diversity; [1] based on weighted Burden matrix [2,3]. "
                "[1] https://doi.org/10.1021/ci980137x "
                "[2] https://doi.org/10.1021/ci00063a011 "
                "[3] https://doi.org/10.1002/qsar.19970160406",
            ),
            (
                "bond_count",
                "BondCountDescriptor",
                1,
                "number of bonds. "
                "has an order: str parameter. "
                "could be varied to count by bond type. ",
            ),
            (
                "bpol",
                "BPolDescriptor",
                1,
                "sum of absolute differences between atomic polarizabilities of all bonded atoms "
                "(including implicit hydrogens); assumes 2-centered bonds. ",
            ),
            (
                "carbon_types",
                "CarbonTypesDescriptor",
                9,
                "C1SP1, C2SP1, C1SP2, C2SP2, C3SP2, C1SP3, C2SP3, C3SP3, C4SP3",
            ),
            (
                "chi_chain",
                "ChiChainDescriptor",
                10,
                "simple and valence chi chain descriptors of orders 3, 4, 5, 6 and 7 "
                "(simple chain 3, 4, 5, 6, 7, valence chain 3, 4, 5, 6, 7). "
                "solves graph isomorphism subproblems for fragment identification, and thus might be slow. "
                "recent versions of Molconn-Z use simplified fragment definitions (i.e., rings without branches etc.) "
                "these descriptors use the older more complex fragment definitions. ",
            ),
            (
                "chi_cluster",
                "ChiClusterDescriptor",
                8,
                "based on simple and valence chi chain descriptors of orders 3, 4, 5, 6 "
                "(simple 3, 4, 5, 6, valence 3, 4, 5, 6). "
                "solves graph isomorphism subproblems for fragment identification, and thus might be slow. "
                "recent versions of Molconn-Z use simplified fragment definitions (i.e., rings without branches etc.) "
                "these descriptors use the older more complex fragment definitions. ",
            ),
            (
                "chi_path",
                "ChiPathDescriptor",
                16,
                "simple paths orders 0, 1, ..., 7, valence paths orders 0, 1, ..., 7. "
                "solves graph isomorphism subproblems for fragment identification, and thus might be slow. "
                "recent versions of Molconn-Z use simplified fragment definitions (i.e., rings without branches etc.) "
                "these descriptors use the older more complex fragment definitions. ",
            ),
            (
                "chi_path_cluster",
                "ChiPathClusterDescriptor",
                6,
                "based on simple and valence chi chain descriptors of orders 4, 5, 6 "
                "(simple 4, 5, 6, valence 4, 5, 6). "
                "solves graph isomorphism subproblems for fragment identification, and thus might be slow. "
                "recent versions of Molconn-Z use simplified fragment definitions (i.e., rings without branches etc.) "
                "these descriptors use the older more complex fragment definitions. ",
            ),
            (
                "cpsa",
                "CPSADescriptor",
                29,
                "charged polar surface area descriptors. "
                "https://doi.org/10.1021/ac00220a013 "
                "this implementation differs from the original ADAPT software implementation",
            ),
            (
                "eccentric_connectivity_index",
                "EccentricConnectivityIndexDescriptor",
                1,
                "topological descriptor combining distance and adjacency information. "
                "https://doi.org/10.1021/ci960049h",
            ),
            (
                "fmf",
                "FMFDescriptor",
                1,
                "ratio of heavy atoms in molecular framework and whole molecule. "
                "https://doi.org/10.1021/jm1008456",
            ),
            (
                "fractional_csp3",
                "FractionalCSP3Descriptor",
                1,
                "characterizes non-flatness. " "https://doi.org/10.1021/jm901241e",
            ),
            (
                "fractional_psa",
                "FractionalPSADescriptor",
                1,
                "polar surface area expressed as ratio to molecular size. ",
            ),
            (
                "fragment_complexity",
                "FragmentComplexityDescriptor",
                1,
                "https://doi.org/10.1021/ci050521b",
            ),
            (
                "gravitational_index",
                "GravitationalIndexDescriptor",
                9,
                "descriptors of mass distribution in molecule [1], as well as squares and cubic roots [2] of these. "
                "[1] https://doi.org/10.1021/jp953224q [2] https://doi.org/10.1021/ci980029a",
            ),
            (
                "h_bond_acceptor_count",
                "HBondAcceptorCountDescriptor",
                1,
                "number of hydrogen bond acceptors, calculated using a simplified version of PHACIR atom types. ",
            ),
            (
                "h_bond_donor_count",
                "HBondDonorCountDescriptor",
                1,
                "number of hydrogen bond donors, calculated using a simplified version of PHACIR atom types. ",
            ),
            (
                "hybridization_ratio",
                "HybridizationRatioDescriptor",
                1,
                "fraction of sp3 carbons to sp3+sp2 carbons. ",
            ),
            (
                "ip_molecular_learning",
                "IPMolecularLearningDescriptor",
                1,
                "ionization potential as predicted by decision trees. "
                "has a addlp: bool (add lone pairs) parameter. "
                "assumes explicit hydrogens. "
                "only handles Cl, Br, I, N, P, O, S, and not for conjugated systems or adjacent to a double bond. ",
            ),
            (
                "jp_log",
                "JPlogPDescriptor",
                1,
                "log P model based on atom contributions. "
                "https://doi.org/10.1186/s13321-018-0316-5",
            ),
            (
                "kappa_shape_indices",
                "KappaShapeIndicesDescriptor",
                3,
                "Kier and Hall kappa molecular shape indices. "
                "compare the molecular graph with minimal and maximal molecular graphs. "
                "https://doi.org/10.1002/9780470125793.ch9",
            ),
            (
                "kier_hall_smarts",
                "KierHallSmartsDescriptor",
                79,
                "e-state fragment counts. "
                "assumes atom typing and aromaticity detection. "
                "https://doi.org/10.3390/91201004",
            ),
            (
                "largest_chain",
                "LargestChainDescriptor",
                1,
                "number of atoms in the longest chain. " "has a checkRingSystem: bool parameter. ",
            ),
            (
                "largest_pi_system",
                "LargestPiSystemDescriptor",
                1,
                "has a checkAromaticity:bool parameter. ",
            ),
            (
                "length_over_breadth",
                "LengthOverBreadthDescriptor",
                2,
                "maximum length/breadth ration and ratio for minimum-area rotation. ",
            ),
            (
                "longest_aliphatic_chain",
                "LongestAliphaticChainDescriptor",
                1,
                "number of atoms in longest aliphatic chain. "
                "has a checkRingSystem: bool parameter. ",
            ),
            (
                "mannhold_logp",
                "MannholdLogPDescriptor",
                1,
                "predicted molecular lipophilicity logP based on number of carbon and hetero atoms. "
                "https://doi.org/10.1002/jps.21494",
            ),
            (
                "mde",
                "MDEDescriptor",
                19,
                "molecular distance edge. "
                "10 original descriptors and variants for O and N "
                "https://doi.org/10.1021/ci970109z",
            ),
            (
                "moment_of_inertia",
                "MomentOfInertiaDescriptor",
                7,
                "moments of inertia (x, y, z axis, x/y, x/z, y/z ratios, radius of gyration. ",
            ),
            (
                "petitjean_number",
                "PetitjeanNumberDescriptor",
                1,
                "edge eccentricity via distance matrix. " "https://doi.org/10.1021/ci00008a012",
            ),
            (
                "petitjean_shape_index",
                "PetitjeanShapeIndexDescriptor",
                2,
                "topological [1] and geometric [2] shape indices. "
                "[1] https://doi.org/10.1021/ci00008a012 [2] https://doi.org/10.1021/ci00026a007",
            ),
            (
                "rotatable_bonds_count",
                "RotatableBondsCountDescriptor",
                1,
                "has a includeTerminals: bool parameter. " "has a excludeAmides: bool parameter. ",
            ),
            (
                "rule_of_five",
                "RuleOfFiveDescriptor",
                1,
                "number of violations of Lipinski's rule of five for drug-likeness. "
                "https://doi.org/10.1016/S0169-409X(00)00129-0",
            ),
            (
                "small_ring",
                "SmallRingDescriptor",
                11,
                "based on enumeration of all small rings sizes 3 to 9. ",
            ),
            (
                "spiro_atom_count",
                "SpiroAtomCountDescriptor",
                1,
                "number of spiro atoms (atoms common to two rings). ",
            ),
            (
                "tpsa",
                "TPSADescriptor",
                1,
                "topological polarizable surface area" "https://doi.org/10.1021/jm000942e",
            ),
            (
                "vabc_volume",
                "VABCDescriptor",
                1,
                "van der Waals volume based on atom and bond counts. "
                "limited to H, C, N, O, F, Cl, Br, I, P, S, As, B, Si, Se, Te. "
                "https://doi.org/10.1021/jo034808o",
            ),
            (
                "v_adj_ma",
                "VAdjMaDescriptor",
                1,
                "vertex adjacency magnitude; 1 + log of number of heavy-heavy bonds. ",
            ),
            (
                "weight",
                "WeightDescriptor",
                1,
                "sum of all atomic weights (molecular weight). "
                "has an elementSymbol: str parameter. "
                "could be varied to count by atom type. ",
            ),
            (
                "weighted_path",
                "WeightedPathDescriptor",
                5,
                "molecular id, molecular id/number of atoms, sum of path lengths "
                "starting from heteroatoms, oxygens, nitrogens. "
                "NP-hard, can be slow for large molecules. "
                "https://doi.org/10.1021/ci00043a009",
            ),
            (
                "whim",
                "WHIMDescriptor",
                17,
                "weighted holistic invariant molecular (WHIM) descriptors. "
                "5 different weighting schemes can be used: unit weights (default), atomic masses, "
                "van der Waals volumes, Mulliken atomic electronegativites, atomic polarizabilities"
                "has a type: int parameter. "
                "could be varied to enable use of different weighting schemes. "
                "https://doi.org/10.1080/10629369708039126",
            ),
            (
                "wiener_numbers",
                "WienerNumbersDescriptor",
                2,
                "Wiener path and polarity numbers. " "https://doi.org/10.1021/ja01193a005",
            ),
            (
                "xlogp",
                "XLogPDescriptor",
                1,
                "logP predicted by XLogP atom-type method. "
                "requires explicit hydrogens. "
                "see also comments on the CDK API page. "
                "has checkAromaticity: bool and salicylFlag: bool parameters. "
                "https://doi.org/10.1021/ci960169p https://doi.org/10.1023/A:1008763405023",
            ),
            (
                "zagreb_index",
                "ZagrebIndexDescriptor",
                1,
                "sum of the squares of atom degree over all heavy atoms. ",
            ),
        )
    }

    PRESET_ALL = tuple(DESCRIPTORS.keys())

    # a subset of descriptors that are fast to compute and do not fail often
    # (tested on QM9 and CEP datasets)
    PRESET_ROBUST = (
        "acidic_group_count",
        "alogp",
        "apol",
        "aromatic_atoms_count",
        "aromatic_bonds_count",
        "atom_count",
        # 'auto_correlation_charge',  # NaN
        "auto_correlation_mass",
        "auto_correlation_polarizability",
        "basic_group_count",
        # 'bcut',  # NaN
        "bond_count",
        "bpol",
        "carbon_types",
        # 'chi_chain',  # too slow
        # 'chi_cluster',  # too slow
        # 'chi_path',  # too slow
        # 'chi_path_cluster',  # too slow
        # 'cpsa',  # NaN
        "eccentric_connectivity_index",
        "fmf",
        "fractional_csp3",
        "fractional_psa",
        "fragment_complexity",
        # 'gravitational_index',  # NaN
        "h_bond_acceptor_count",
        "h_bond_donor_count",
        "hybridization_ratio",
        # 'ip_molecular_learning',  # NaN
        "jp_log",
        "kappa_shape_indices",
        "kier_hall_smarts",
        "largest_chain",
        "largest_pi_system",
        # 'length_over_breadth',  # NaN
        # 'longest_aliphatic_chain',  # fails
        "mannhold_logp",
        "mde",
        # 'moment_of_inertia',  # NaN
        "petitjean_number",
        # 'petitjean_shape_index',  # NaN
        "rotatable_bonds_count",
        "rule_of_five",
        "small_ring",
        "spiro_atom_count",
        "tpsa",
        # 'vabc_volume',  # NaN
        "v_adj_ma",
        "weight",
        # 'weighted_path',  # too slow
        # 'whim',  # NaN
        "wiener_numbers",
        "xlogp",
        "zagreb_index",
    )

    # py4j Java gateway process for CDK molecular features
    # implemented as a class attribute to keep the gateway alive across different
    # instantiations of the class as gateway creation is somewhat expensive (starts new process)
    _java_gateway = None

    def __init__(
        self,
        select: Optional[Sequence[str]] = None,
        failmode="raise",
        samplef: Callable[[Any], Any] = lambda arg: arg,
        java_gateway: Optional[CdkJavaGateway] = None,
        **kwargs,
    ):
        """Initialize state.

        Parameters:
            select: which features to compute (by default, all). List of names, order matters.
                Presets are available as class constants:
                PRESET_ALL: all features
                PRESET_ROBUST: a subset of descriptors that are fast to compute and do not fail
                    often (tested on QM9 and CEP datasets; see accompanying notebook)
            failmode: how to handle failed descriptor calculations, either due to rejected SMILES
                encodings or failing descriptor code. Possible values:
                "raise" [default]: raise a Benchmarexception
                "drop": drop the sample. Returned Data will have fewer samples
                ("mask", mask): where `mask` is a NumPy array with dtype bool whose entries will
                    be set to False for failures
                ("index", index): where `index` is an empty list to which the indices of failed
                    entries will be appended
            samplef: a function accepting and returning a sample. This enables
                transformation of samples, for example, to select an entry by key
                if sample is a dictionary, or to turn a dictionary into a vector.
                Default is to return the sample unchanged.
            java_gateway: a gateway to a Java virtual machine

        Requires a CDK jar.
        """

        super().__init__(**kwargs)

        # parameters
        select = params.optional_(
            select,
            lambda arg: params.tuple_(
                arg, lambda arg: params.enumeration(arg, self.DESCRIPTORS.keys())
            ),
        )
        select = self.PRESET_ALL if select is None else select
        self._failmode = DataTransformationFailureMode.failmode(failmode)
        self._samplef = params.callable(samplef, num_pos_or_kw=1)
        self._java_gateway = params.optional_(
            java_gateway, lambda arg: params.instance(arg, JavaGateway)
        )
        if self._java_gateway is None:
            self._java_gateway = CdkJavaGateway()
        self._java_gateway = self._java_gateway.gateway

        # set up descriptors
        self._descriptors = tuple(
            eval("self._java_gateway.jvm." + self.DESCRIPTORS[name][0] + "()") for name in select
        )

        builder = self._java_gateway.jvm.org.openscience.cdk.DefaultChemObjectBuilder.getInstance()
        for descriptor in self._descriptors:
            descriptor.initialise(builder)

        self._arities = tuple(self.DESCRIPTORS[name][1] for name in select)

    def apply(self, data: Data) -> TabularData:
        """Compute selected molecular features.

        Parameters:
            data: molecular structures given as SMILES strings.
                  Can be labeled, and labels will be retained

        Returns:
            TabularData with CDK molecular features as samples
        """

        data = params.instance(data, Data)  # todo: params.data(data, is_finite=True)

        failmode = DataTransformationFailureMode(self._failmode, data.num_samples)

        # set up molecule SMILES
        builder = self._java_gateway.jvm.org.openscience.cdk.DefaultChemObjectBuilder.getInstance()
        parser = self._java_gateway.jvm.org.openscience.cdk.smiles.SmilesParser(builder)

        def parse_smiles(s: str, i: int):
            """Return parsed SMILES string or None on failure."""
            try:
                return parser.parseSmiles(self._samplef(s))
            except py4j.protocol.Py4JJavaError:
                # expected to be raised from org.openscience.cdk.exception.InvalidSmilesException
                failmode.handle_failure(i)
                return None  # internal sentinel value

        smiles = tuple(parse_smiles(s, i) for i, s in enumerate(data.samples()))

        # compute descriptors
        # todo: the dtype of the columns could be set in advance by querying the descriptors
        #       currently, all values are stored as floating point numbers
        features = np.empty((data.num_samples, np.sum(self._arities)))
        index = 0

        def java_is_instance_of(object_, class_):
            return py4j.java_gateway.is_instance_of(
                self._java_gateway, object_, "org.openscience.cdk.qsar.result." + class_
            )

        def check_arity(expected, actual):
            if expected != actual:
                raise BenchmarkError(
                    f"Invalid descriptor result arity (expected {expected}, was {actual})"
                )

        for descriptor, arity in zip(self._descriptors, self._arities):
            for i, smile in enumerate(smiles):
                if smiles is None:
                    features[i, index : index + arity] = float("nan")
                    continue

                try:
                    value = descriptor.calculate(smile).getValue()
                except py4j.protocol.Py4JJavaError:
                    failmode.handle_failure(i)
                    features[i, index : index + arity] = float("nan")
                    continue

                if java_is_instance_of(value, "IntegerResult"):
                    check_arity(arity, 1)
                    features[i, index] = int(value.intValue())
                elif java_is_instance_of(value, "DoubleResult"):
                    check_arity(arity, 1)
                    features[i, index] = float(value.doubleValue())
                elif java_is_instance_of(value, "BooleanResult"):
                    check_arity(arity, 1)
                    features[i, index] = bool(value.booleanValue())
                elif java_is_instance_of(value, "IntegerArrayResult"):
                    check_arity(arity, value.length())
                    features[i, index : index + arity] = tuple(
                        int(value.get(j)) for j in range(value.length())
                    )
                elif java_is_instance_of(value, "DoubleArrayResult"):
                    check_arity(arity, value.length())
                    features[i, index : index + arity] = tuple(
                        float(value.get(j)) for j in range(value.length())
                    )
                # there seems to be no BooleanArrayResult in CDK
                else:
                    name = value.getClass().getSimpleName()
                    raise BenchmarkError(f"Unsupported CDK result type '{name}'")
            index += arity

        result = (
            TabularData(data=features, labels=data.labels())
            if data.is_labeled
            else TabularData(data=features)
        )

        result = failmode.finalize(result)

        return result


#  .-------------------.
#  |                   |
#  |  A P P E N D I X  |
#  |                   |
#  .-------------------.

# Contains:
#   Excluded descriptors
#   List of all descriptors in CDK
#   List of all fingerprints in CDK


# Excluded descriptors

# AminoAcidCountDescriptor                # not relevant for organic molecules
# AtomDegreeDescriptor                    # atom descriptor
# AtomHybridizationDescriptor             # atom descriptor
# AtomHybridizationVSEPRDescriptor        # atom descriptor
# AtomicNumberDifferenceDescriptor        # bond descriptor
# AtomValenceDescriptor                   # atom descriptor
# BondPartialPiChargeDescriptor           # bond descriptor
# BondPartialSigmaChargeDescriptor        # bond descriptor
# BondPartialTChargeDescriptor            # bond descriptor
# BondSigmaElectronegativityDescriptor    # bond descriptor
# BondsToAtomDescriptor                   # atom descriptor
# CovalentRadiusDescriptor                # atom descriptor
# DistanceToAtomDescriptor                # atom descriptor
# EffectiveAtomPolarizabilityDescriptor   # atom descriptor
# InductiveAtomicHardnessDescriptor       # atom descriptor
# InductiveAtomicSoftnessDescriptor       # atom descriptor
# IPAtomicHOSEDescriptor                  # atom descriptor
# IPAtomicLearningDescriptor              # atom descriptor
# IPBondLearningDescriptor                # bond descriptor
# IsProtonInAromaticSystemDescriptor      # atom descriptor
# IsProtonInConjugatedPiSystemDescriptor  # atom descriptor
# OxygenAtomCountDescriptor               # substance descriptor
# PartialPiChargeDescriptor               # atom descriptor
# PartialSigmaChargeDescriptor            # bond descriptor
# PartialTChargeMMFF94Descriptor          # atom descriptor
# PartialTChargePEOEDescriptor            # atom descriptor
# PeriodicTablePositionDescriptor         # atom descriptor
# PiContactDetectionDescriptor            # atom pair descriptor
# PiElectronegativityDescriptor           # atom descriptor
# ProtonAffinityHOSEDescriptor            # atom descriptor
# ProtonTotalPartialChargeDescriptor      # atom descriptor
# RDFProtonDescriptor_G3R                 # atom descriptor
# RDFProtonDescriptor_GDR                 # atom descriptor
# RDFProtonDescriptor_GHR                 # atom descriptor
# RDFProtonDescriptor_GHR_topol           # atom descriptor
# RDFProtonDescriptor_GSR                 # atom descriptor
# SigmaElectronegativityDescriptor        # atom descriptor
# StabilizationPlusChargeDescriptor       # atom descriptor
# VdWRadiusDescriptor                     # atom descriptor


# Complete list of all 93 ...Descriptor classes in CDK v2.3:

# AcidicGroupCountDescriptor
# ALOGPDescriptor
# AminoAcidCountDescriptor
# APolDescriptor
# AromaticAtomsCountDescriptor
# AromaticBondsCountDescriptor
# AtomCountDescriptor
# AtomDegreeDescriptor
# AtomHybridizationDescriptor
# AtomHybridizationVSEPRDescriptor
# AtomicNumberDifferenceDescriptor
# AtomValenceDescriptor
# AutocorrelationDescriptorCharge
# AutocorrelationDescriptorMass
# AutocorrelationDescriptorPolarizability
# BasicGroupCountDescriptor
# BCUTDescriptor
# BondCountDescriptor
# BondPartialPiChargeDescriptor
# BondPartialSigmaChargeDescriptor
# BondPartialTChargeDescriptor
# BondSigmaElectronegativityDescriptor
# BondsToAtomDescriptor
# BPolDescriptor
# CarbonTypesDescriptor
# ChiChainDescriptor
# ChiClusterDescriptor
# ChiPathClusterDescriptor
# ChiPathDescriptor
# CovalentRadiusDescriptor
# CPSADescriptor
# DistanceToAtomDescriptor
# EccentricConnectivityIndexDescriptor
# EffectiveAtomPolarizabilityDescriptor
# FMFDescriptor
# FractionalCSP3Descriptor
# FractionalPSADescriptor
# FragmentComplexityDescriptor
# GravitationalIndexDescriptor
# HBondAcceptorCountDescriptor
# HBondDonorCountDescriptor
# HybridizationRatioDescriptor
# InductiveAtomicHardnessDescriptor
# InductiveAtomicSoftnessDescriptor
# IPAtomicHOSEDescriptor
# IPAtomicLearningDescriptor
# IPBondLearningDescriptor
# IPMolecularLearningDescriptor
# IsProtonInAromaticSystemDescriptor
# IsProtonInConjugatedPiSystemDescriptor
# JPlogPDescriptor
# KappaShapeIndicesDescriptor
# KierHallSmartsDescriptor
# LargestChainDescriptor
# LargestPiSystemDescriptor
# LengthOverBreadthDescriptor
# LongestAliphaticChainDescriptor
# MannholdLogPDescriptor
# MDEDescriptor
# MomentOfInertiaDescriptor
# OxygenAtomCountDescriptor
# PartialPiChargeDescriptor
# PartialSigmaChargeDescriptor
# PartialTChargeMMFF94Descriptor
# PartialTChargePEOEDescriptor
# PeriodicTablePositionDescriptor
# PetitjeanNumberDescriptor
# PetitjeanShapeIndexDescriptor
# PiContactDetectionDescriptor
# PiElectronegativityDescriptor
# ProtonAffinityHOSEDescriptor
# ProtonTotalPartialChargeDescriptor
# RDFProtonDescriptor_G3R
# RDFProtonDescriptor_GDR
# RDFProtonDescriptor_GHR
# RDFProtonDescriptor_GHR_topol
# RDFProtonDescriptor_GSR
# RotatableBondsCountDescriptor
# RuleOfFiveDescriptor
# SigmaElectronegativityDescriptor
# SmallRingDescriptor
# SpiroAtomCountDescriptor
# StabilizationPlusChargeDescriptor
# TPSADescriptor
# VABCDescriptor
# VAdjMaDescriptor
# VdWRadiusDescriptor
# WeightDescriptor
# WeightedPathDescriptor
# WHIMDescriptor
# WienerNumbersDescriptor
# XLogPDescriptor
# ZagrebIndexDescriptor


# List of all ...Fingerprint(er) classes in CDK v2.3:

# this list has not been processed yet
# entries marked with "unclear" are likely not actual fingerprints but infrastructure classes

# AtomPairs2DFingerprinter
# BitSetFingerprint                 # unclear
# CircularFingerprinter             # unclear
# CircularFingerprinter.FP          # unclear
# EStateFingerprinter
# ExtendedFingerprinter             # unclear
# Fingerprinter                     # unclear
# FingerprinterTool                 # unclear
# FingerprintFormat                 # unclear
# GraphOnlyFingerprinter            # unclear
# HybridizationFingerprinter        # unclear
# IntArrayCountFingerprint          # unclear
# IntArrayFingerprint               # unclear
# KlekotaRothFingerprinter
# LingoFingerprinter
# MACCSFingerprinter
# PubchemFingerprinter
# ShortestPathFingerprinter
# SignatureFingerprinter
# SubstructureFingerprinter
