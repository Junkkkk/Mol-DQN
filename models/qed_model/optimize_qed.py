"""Optimizes QED of a molecule with DQN.
This experiment tries to find the molecule with the highest QED
starting from a given molecule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem

from models import deep_q_networks, trainer, molecules_mdp



class QEDRewardMolecule(molecules_mdp.Molecule_MDP):
    def __init__(self, molecules, **kwargs):
        """Initializes the class.
        Args:
          discount_factor: Float. The discount factor. We only
          care about the molecule at the end of modification.
          In order to prevent a myopic decision, we discount
          the reward at each step by a factor of
          discount_factor ** num_steps_left,
          this encourages exploration with emphasis on long term rewards.
        **kwargs: The keyword arguments passed to the base class.
        """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.molecules = molecules

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

    def _reward(self):
        """Reward of a state.
        Returns:
          Float. QED of the current state.
        """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)