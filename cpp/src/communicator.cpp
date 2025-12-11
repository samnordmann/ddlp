#include "ddlp/communicator.hpp"
#include <stdexcept>
#include <iostream>

namespace ddlp {

CommunicatorImpl::CommunicatorImpl() : initialized_mpi_(false) {
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        initialized_mpi_ = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

CommunicatorImpl::~CommunicatorImpl() {
    // Note: Usually we don't finalize if we didn't init, or if we want to allow others to continue.
    // For this simple scaffolding, we leave it be or could add a finalize check.
    if (initialized_mpi_) {
        // MPI_Finalize(); // Be careful finalizing in library code
    }
}

int CommunicatorImpl::rank() const {
    return rank_;
}

int CommunicatorImpl::world_size() const {
    return size_;
}

void CommunicatorImpl::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace ddlp



