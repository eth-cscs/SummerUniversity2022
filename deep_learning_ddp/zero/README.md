# Understanding the effect of ZeRo-{1, 2, 3} on memory

Here we are going to check the memory consumption of the different stages of ZeRo.
This is based on PyTorch's example [Shard optimizer states with `ZeroRedundancyOptimizer`](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html).

In the script [`pt_deepspeed_check_mem.py`](pt_deepspeed_check_mem.py) we define a simple model with different number of parameters and measure the total allocated memory on the GPU at different points of the execution.

```bash
srun --pty  python pt_deepspeed_check_mem.py --deepspeed_config ds_config.json --data-dim 10000
```
