"""Optimizes QED of a molecule with DQN.
This experiment tries to find the molecule with the highest QED
starting from a given molecule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED


from models import molecules_mdp

class Multi_LogP_QED_Molecule(molecules_mdp.Molecule_MDP):
    """
    Defines the subclass of generating a molecule with a specific reward.
    The reward is defined as a scalar
    reward = weight * similarity_score + (1 - weight) *  qed_score
    """

    def __init__(self, molecules, **kwargs):
        """Initializes the class.
        Args:
          target_molecule: SMILES string. The target molecule against which we
            calculate the similarity.
          similarity_weight: Float. The weight applied similarity_score.
          discount_factor: Float. The discount factor applied on reward.
          **kwargs: The keyword arguments passed to the parent class.
        """
        super(Multi_LogP_QED_Molecule, self).__init__(**kwargs)
        self._all_molecules = molecules
        self._sim_weight = self.hparams["train_param"]["similarity_weight"]

    def initialize(self):
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

    def get_similarity(self, smiles):
        """Gets the similarity between the current molecule and the target molecule.
        Args:
          smiles: String. The SMILES string for the current molecule.
        Returns:
          Float. The Tanimoto similarity.
        """
        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                              fingerprint_structure)

    def _reward(self):
        """Calculates the reward of the current state.
        The reward is defined as a tuple of the similarity and QED value.
        Returns:
          A tuple of the similarity and qed value
        """
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0
        mol = Chem.MolFromSmiles(self._state)
        if mol is None:
            return 0.0
        similarity_score = self.get_similarity(self._state)
        # calculate QED
        qed_value = QED.qed(mol)
        reward = (
            similarity_score * self._sim_weight +
            qed_value * (1 - self._sim_weight))
        discount = self.discount_factor**(self.max_steps - self._counter)

        return reward * discount