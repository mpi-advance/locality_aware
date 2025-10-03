// GPU Method to find where memory was allocated
#ifdef GPU

void get_memcpy_kind(gpuMemoryType send_type,
                     gpuMemoryType recv_type,
                     gpuMemcpyKind* memcpy_kind)
{
    if (send_type == gpuMemoryTypeDevice && recv_type == gpuMemoryTypeDevice)
    {
        *memcpy_kind = gpuMemcpyDeviceToDevice;
    }
    else if (send_type == gpuMemoryTypeDevice)
    {
        *memcpy_kind = gpuMemcpyDeviceToHost;
    }
    else if (recv_type == gpuMemoryTypeDevice)
    {
        *memcpy_kind = gpuMemcpyHostToDevice;
    }
    else
    {
        *memcpy_kind = gpuMemcpyHostToHost;
    }
}
#endif