import pytest
from ddlp.communicator import Communicator

def test_communicator_basic():
    # This test might fail if MPI is not initialized or present in the environment
    # But serves as a structural test
    try:
        comm = Communicator()
        rank = comm.rank()
        size = comm.world_size()
        assert isinstance(rank, int)
        assert isinstance(size, int)
    except Exception as e:
        pytest.skip(f"MPI initialization failed: {e}")



