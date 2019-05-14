from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/home/junyoung/workspace/Lead_Optimization')

import functools
import json

from absl import app
from models import deep_q_networks, trainer
from models.qed_model.optimize_qed import QEDRewardMolecule


def main(argv):
    del argv  # unused.
    config_name = '/home/junyoung/workspace/Lead_Optimization/Config/naive_dqn'
    all_cid = '/home/junyoung/workspace/Lead_Optimization/Config/all_cid'

    with open(config_name) as f:
        hparams = json.load(f)

    with open(all_cid) as f:
        all_mols = json.load(f)


    environment = QEDRewardMolecule(hparams=hparams,
                                    init_mol=None,
                                    all_molecules=all_mols)

    dqn = deep_q_networks.DeepQNetwork(
        hparams=hparams,
        q_fn=functools.partial(
            deep_q_networks.multi_layer_model, hparams=hparams),
        grad_clipping=None)

    Trainer =trainer.Trainer(
        hparams=hparams,
        environment=environment,
        dqn=dqn)

    Trainer.run_training()

if __name__ == '__main__':
    app.run(main)