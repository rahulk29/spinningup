# Algorithms
from spinup.algos.ddpg.ddpg import ddpg
from spinup.algos.ddpg.ddpg_torch import ddpg as ddpg_torch
from spinup.algos.priority_ddpg.ddpg import ddpg as priority_ddpg
from spinup.algos.psn_ddpg.ddpg import ddpg as psn_ddpg
from spinup.algos.rdpg.rdpg import rdpg
from spinup.algos.ppo.ppo import ppo
from spinup.algos.sac.sac import sac
from spinup.algos.td3.td3_randtarg import td3 as td3_randtarg
from spinup.algos.td3.td3 import td3
from spinup.algos.per_td3.td3 import td3 as per_td3
from spinup.algos.trpo.trpo import trpo
from spinup.algos.vpg.vpg import vpg
from spinup.algos.dqn.dqn import dqn
from spinup.algos.ddpg_ens_per.ddpg_torch import ddpg_ens_per
# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__