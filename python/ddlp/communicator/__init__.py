import torch.distributed as dist


class Communicator:
    """Thin wrapper around torch.distributed for rank/world_size/barrier."""

    def __init__(self, process_group=None):
        self._pg = process_group

    def rank(self):
        return dist.get_rank(self._pg)

    def world_size(self):
        return dist.get_world_size(self._pg)

    def barrier(self):
        dist.barrier(group=self._pg)
