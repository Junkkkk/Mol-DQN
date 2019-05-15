from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import sys
sys.path.append('/home/junyoung/workspace/Mol_DQN')

import os

from Config import config
from models import deep_q_networks, trainer


from models.logp_model.optimize_logp import LogP_Molecule


def main():
    config_name = '/home/junyoung/workspace/Mol_DQN/models/logp_model/config'
    all_cid = '/home/junyoung/workspace/Mol_DQN/Config/all_cid'

    with open(config_name) as f:
        hparams = json.load(f)

    # with open(all_cid) as f:
    #     all_mols = json.load(f)


    environment = LogP_Molecule(hparams=hparams, init_mol=None)

    dqn = deep_q_networks.DeepQNetwork(
        hparams=hparams,
        q_fn=functools.partial(
            deep_q_networks.Q_fn_neuralnet_model, hparams=hparams))

    Trainer =trainer.Trainer(
        hparams=hparams,
        environment=environment,
        model=dqn)

    Trainer.run_training()

    config.write_hparams(hparams, os.path.join(hparams['save_param']['model_dir'], 'config.json'))


if __name__ == '__main__':
    main()