#ifndef MPIL_UTILS_H
#define MPIL_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif  // General utility methods (that use C++ functions)

/** @brief wrapper around std::sort
        @param [in] n_objects number of objects to short
        @param [in, out] array of indexes
        @param [in] array of values
**/
void sort(int n_objects, int* object_indices, int* object_values);

/** @brief wrapper around std::rotate,
 *	@details
 *      Rotates such that new_first_byte is first in array
 *		Divides recvbuf into two parts [first, middle] and (middle, last)
 *		then swaps their positioning.
 *		Example: A = 0, 1, 2, 3, 4, 5
 *           std::rotate(A*, 2, A*+6) would split into (0, 1) and (2, 3, 4, 5)
 *			 and after running A = 2, 3, 4, 5, 0, 1
 *
 *	@param [in, out] recvbuf buffer of elements to rotate
 *	@param [in] new_first_byte index immediately after the split point.
 *	@param [in] index of last element in the effected range
 **/
void rotate(void* ref, int new_start_byte, int end_byte);

/** @brief reverses order of elements in recv_buffer
 *	@details
 *		Divides recvbuf into two parts [first, middle] and (middle, last)
 *		then swaps their positioning.
 *		Example: A = 0, 1, 2, 3, 4, 5
 *           std::rotate(A*, 2, A*+6) would split into (0, 1) and (2, 3, 4, 5)
 *			 and after running A = 2, 3, 4, 5, 0, 1
 *
 *	@param [in, out] recvbuf buffer of elements to rotate
 *	@param [in] new_first_byte index immediately after the split point.
 *	@param [in] index of last element in the effected range
 *  \todo why this instead of std::reverse?
 **/
void reverse(void* recvbuf, int n_bytes, int var_bytes);

#ifdef __cplusplus
}
#endif

#endif