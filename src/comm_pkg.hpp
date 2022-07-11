#ifndef MPI_ADVANCE_COMM_PKG_HPP
#define MPI_ADVANCE_COMM_PKG_HPP

// Comm Data Structure:
struct CommData
{
    int num_msgs;
    int size_msgs;
    int* procs;
    int* indptr;
    int* indices;
    char* buffer;

    CommData()
    {
        num_msgs = 0;
        size_msgs = 0;
        procs = NULL;
        indptr = NULL;
        indices = NULL;
        buffer = NULL;
    }

    ~CommData()
    {
        if (procs)
            delete[] procs;
        if (indptr)
            delete[] indptr;
        if (indices)
            delete[] indices;
    }

    void init(int _num_msgs)
    {
        num_mmsgs = _num_msgs;
        if (num_msgs)
            procs = new int[num_msgs];
        indptr = new int[num_msgs+1];
        indptr[0] = 0;
    }

    void remove_duplicates()
    {
        int start, end;

        for (int i = 0; i < num_msgs; i++)
        {
            start = indptr[i];
            end = indptr[i+1];
            std::sort(indices+start, indices+end);
        }

        size_msgs = 0;
        start = indptr[0];
        for (int i = 0; i < num_msgs; i++)
        {
            end = indptr[i+1];
            indices[size_msgs++] = indices[start];
            for (int j  = start; j < end - 1; j++)
            {
                if (indices[j+1] != indices[j])
                {
                    indices[size_msgs++] = indices[j+1];
                }
            }
            start = end;
            indptr[i+1] = size_msgs;
        }
    }
};

struct CommPkg
{
    CommData* send_data;
    CommData* recv_data;
    int tag;

    CommPkg(int _tag)
    {
        send_data = new CommData();
        recv_data = new CommData();
        tag = _tag;
    }

    ~CommPkg()
    {
        delete send_data;
        delete recv_data;
    }
};

struct LocalityComm
{
    CommPkg* local_L_comm;
    CommPkg* local_S_comm;
    CommPkg* local_R_comm;
    CommPkg* global_comm;
    Topology* topology;

    LocalityComm(MPI_Comm comm = MPI_COMM_WORLD)
    {
        topology = new Topology(comm);

        local_L_comm = new CommPkg(19234);
        local_S_comm = new CommPkg(92835);
        local_R_comm = new CommPkg(29301);
        global_comm = new CommPkg(72459);
    }

    ~LocalityComm()
    {
        delete local_L_comm;
        delete local_S_comm;
        delete local_R_comm;
        delete global_comm;
    }

};    
    


#endif
