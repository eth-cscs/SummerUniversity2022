import os
import subprocess


def setup_distr_env():
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    node_list = os.environ['SLURM_NODELIST']
    master_node = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1'
    )
    os.environ['MASTER_ADDR'] = master_node
