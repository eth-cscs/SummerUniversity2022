import argparse
import deepspeed
import json
import numpy as np
import torch
import torch.nn as nn
from rich.pretty import pprint


def print_peak_memory(prefix, device):
    max_memory_alloc_mb = torch.cuda.max_memory_allocated(device) // 1e6
    pprint(f"{prefix}: {max_memory_alloc_mb}MB")


# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark')
parser.add_argument('--num-iters', type=int, default=10,
                    help='Number of benchmark iterations')
parser.add_argument('--data-dim', type=int, default=10_000,
                    help='Dimension of the data')
parser.add_argument('--num-layers', type=int, default=20,
                    help='Number of layer in the model')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# FIXME: get the batch_size from config
with open(args.deepspeed_config) as f:
    config_json = json.load(f)

config = deepspeed.runtime.config.DeepSpeedConfig(config_json)
batch_size = config.train_batch_size

local_rank = 0
num_layers = 20


# Set up fixed fake data
class FakeData(torch.utils.data.Dataset):
    def __len__(self):
        return batch_size * args.num_iters

    def __getitem__(self, idx):
        return (torch.randn(20, args.data_dim),
                torch.randn(20, args.data_dim))


trainset = FakeData()
trainloader = torch.utils.data.DataLoader(trainset,
                                          num_workers=1)

# create local model
model = nn.Sequential(*[nn.Linear(args.data_dim, args.data_dim)
                        for _ in range(num_layers)])
print_peak_memory("Max memory allocated after creating model", local_rank)

# --data-dim=10000   # 2,000,200,000 parameters
# --data-dim=15000   # 4,500,300,000 parameters
# --data-dim=20000   # 8,000,400,000 parameters
parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in parameters])
pprint(f'{num_params:,} parameters')

# get again the parameters or copy then before printing above
parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=trainset
)
print_peak_memory("Max memory allocated after deepspeed.initialize",
                  local_rank)

# not using `loss_fn = nn.MSELoss().cuda()` as mse_loss layers autocast
# to float32 https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

# training
for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        inputs = data[0].to(model_engine.device)
        labels = data[1].to(model_engine.device)
        if model_engine.fp16_enabled():
            inputs = inputs.half()

        # forward + backward + optimize
        outputs = model_engine(inputs)
        loss = torch.sum(torch.abs(outputs - labels))
        model_engine.backward(loss)
        model_engine.step()
        print_peak_memory("Max memory allocated after optimizer step",
                          local_rank)

print('Finished Training')
