import torch
import os
import torch.distributed as dist

rank=int(os.getenv("RANK"))
world_size=int(os.getenv("WORLD_SIZE"))
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

dist.init_process_group("nccl", rank=rank, world_size=world_size)
tensor = torch.ones([1,1]).cuda()
print(tensor)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(tensor)
