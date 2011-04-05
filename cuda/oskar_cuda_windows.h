#ifndef OSKAR_CUDA_WINDOWS_H_
#define OSKAR_CUDA_WINDOWS_H_

#ifdef _WIN32
    #ifdef oskar_cuda_EXPORTS
        #define DllExport __declspec(dllexport)
    #else
        #define DllExport
    #endif
#else
    #define DllExport
#endif

#endif