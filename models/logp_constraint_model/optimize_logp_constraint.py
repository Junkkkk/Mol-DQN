from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from models import molecules_mdp
from models import molecules_rules

class LogP_SimilarityConstraintMolecule(molecules_mdp.Molecule_MDP):
    """The molecule whose reward is the penalized logP with similarity constraint.
       Each time the environment is initialized, we uniformly choose
       a molecule from all molecules as target.
    """
    def __init__(self, molecules, similarity_constraint, **kwargs):

        """Initializes the class.
            Args:
                all_molecules: List of SMILES string. the molecules to select
                similarity_constraint: Float. The lower bound of similarity of the
                molecule must satisfy.
            **kwargs: The keyword arguments passed to the parent class.
        """
        super(LogP_SimilarityConstraintMolecule, self).__init__(**kwargs)
        self.molecules = molecules
        self._similarity_constraint = similarity_constraint

    def initialize(self):
        """Resets the MDP to its initial state.
        Each time the environment is initialized, we uniformly choose
        a molecule from all molecules as target.
        """
        self._state = random.choice(self.molecules)
        self._target_mol_fingerprint = self.get_fingerprint(
            Chem.MolFromSmiles(self._state))
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0
        return self._state

    def get_fingerprint(self, molecule):
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, molecule):
        """Gets the similarity between the current molecule and the target molecule.
        Args:
          molecule: String. The SMILES string for the current molecule.
        Returns:
          Float. The Tanimoto similarity.
        """
        fingerprint_structure = self.get_fingerprint(molecule)
        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                              fingerprint_structure)

    def _reward(self):
        """Reward of a state.
        If the similarity constraint is not satisfied,
        the reward is decreased by the difference times a large constant
        If the similarity constrain is satisfied,
        the reward is the penalized logP of the molecule.
        Returns:
          Float. The reward.
        Raises:
          ValueError: if the current state is not a valid molecule.
        """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            raise ValueError('Current state %s is not a valid molecule' % self._state)
        similarity = self.get_similarity(molecule)
        if similarity <= self._similarity_constraint:
            # 40 is an arbitrary number. Suppose we have a molecule that is not
            # similar to the target at all, but has a high logP. The logP improvement
            # can be 20, and the similarity difference can be 0.2. To discourage that
            # molecule, similarity difference is timed by 20 / 0.2 = 100.
            reward = molecules_rules.penalized_logp(molecule) + 100 * (
                similarity - self._similarity_constraint)
        else:
            reward = molecules_rules.penalized_logp(molecule)
        return reward * self.discount_factor**(self.max_steps - self._counter)