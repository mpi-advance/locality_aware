#include "../../include/utils/utils.h"
#include <algorithm>


void rotate(void* recvbuf, int new_first_byte, int last_byte)
{
    char* recv_buffer = (char*)(recvbuf);
    std::rotate(recv_buffer, &(recv_buffer[new_first_byte]), &(recv_buffer[last_byte]));
}