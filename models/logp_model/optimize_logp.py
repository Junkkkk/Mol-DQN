"""Optimize the logP of a molecule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem
import random

from models import molecules_mdp
from models import molecules_rules

class LogP_Molecule(molecules_mdp.Molecule_MDP):
    def __init__(self, all_molecules, **kwargs):
        super(LogP_Molecule, self).__init__(**kwargs)
        self._all_molecules = all_molecules

    def initialize(self):
        self._state = random.choice(self._all_molecules)
        self._target_mol_fingerprint = self.get_fingerprint(
          Chem.MolFromSmiles(self._state))
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0

    def get_fingerprint(self, molecule):
        return Chem.AllChem.GetMorganFingerprint(molecule, radius=2)

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return molecules_rules.penalized_logp(molecule)