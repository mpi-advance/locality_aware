#ifndef MPI_ADVANCE_UTILS_H
#define MPI_ADVANCE_UTILS_H

#ifdef __cplusplus
extern "C"
{
#endif

void sort(int n_objects, int* object_indices, int* object_values);
void rotate(void* ref, int new_start_byte, int end_byte);

#ifdef __cplusplus
}
#endif


#endif
