#include "../../include/utils/utils.h"
#include <algorithm>


void reverse(void* recvbuf, int n_bytes, int var_bytes)
{
    char* recv_buffer = (char*)(recvbuf);
    int n_vars        = n_bytes / var_bytes;
    for (int i = 0; i < n_vars / 2; i++)
    {
        for (int j = 0; j < var_bytes; j++)
        {
            std::swap(recv_buffer[i * var_bytes + j],
                      recv_buffer[(n_vars - i - 1) * var_bytes + j]);
        }
    }
}