"""Optimize the logP of a molecule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem

from models import molecules_mdp
from models import molecules_rules

class LogP_Molecule(molecules_mdp.Molecule_MDP):
    def __init__(self, all_molecules, **kwargs):
        super(LogP_Molecule, self).__init__(**kwargs)
        self._all_molecules = all_molecules

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        logp = molecules_rules.penalized_logp(molecule)
        return logp * self.discount_factor ** (self.max_steps - self.num_steps_taken)