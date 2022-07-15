#include "utils.h"
#include <algorithm>

void sort(int n_objects, int* object_indices, int* object_values)
{
    std::sort(object_indices, object_indices+n_objects,
            [&](const int i, const int j)
            {
                return object_values[i] > object_values[j];
            });
}
