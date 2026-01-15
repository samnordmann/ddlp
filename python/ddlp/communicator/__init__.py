import ddlp._C.communicator as _cpp_communicator

class Communicator:
    """
    DDLP Communicator Component.
    Gathers MPI/Distributed info and manages backend communication contexts.
    """
    def __init__(self):
        self._impl = _cpp_communicator.CommunicatorImpl()
    
    def rank(self):
        return self._impl.rank()
        
    def world_size(self):
        return self._impl.world_size()
    
    def barrier(self):
        self._impl.barrier()

    def finalize(self):
        self._impl.finalize()
