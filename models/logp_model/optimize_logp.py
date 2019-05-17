"""Optimize the logP of a molecule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem

from models import molecules_mdp
from models import molecules_rules

class LogP_Molecule(molecules_mdp.Molecule_MDP):
    def __init__(self, molecules, **kwargs):
        super(LogP_Molecule, self).__init__(**kwargs)
        self.molecules = molecules

    def initialize(self):
        """Resets the MDP to its initial state.
        Each time the environment is initialized, we uniformly choose
        a molecule from all molecules as target.
        """
        self._state = self.molecules
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0
        return self._state

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        logp = molecules_rules.penalized_logp(molecule)
        return logp * self.discount_factor ** (self.max_steps - self.num_steps_taken)