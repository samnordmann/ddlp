#pragma once
#include <mpi.h>
#include <vector>

namespace ddlp {

class CommunicatorImpl {
public:
    CommunicatorImpl();
    ~CommunicatorImpl();
    
    int rank() const;
    int world_size() const;
    void barrier();
    void finalize();

private:
    int rank_;
    int size_;
    bool initialized_mpi_;
};

} // namespace ddlp



