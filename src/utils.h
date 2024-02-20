#ifndef MPI_ADVANCE_UTILS_H
#define MPI_ADVANCE_UTILS_H

#ifdef __cplusplus
extern "C"
{
#endif

void rotate(void* ref, int new_start_byte, int end_byte);
void reverse(void* recvbuf, int n_bytes, int var_bytes);

#ifdef __cplusplus
}
#endif


#endif
