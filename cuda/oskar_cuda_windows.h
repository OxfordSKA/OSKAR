#ifndef OSKAR_CUDA_WINDOWS_H_
#define OSKAR_CUDA_WINDOWS_H_

// Macro used for creating windows the library.
// Note: should only be needed in header files.
//
// The __declspec(dllexport) modifier enables the method to
// be exported by the DLL so that it can be used by other applications.
//
// Usage examples:
//  DllExport void foo();
//  static DllExport double add(double a, double b);
// 
// For more information see:
// http://msdn.microsoft.com/en-us/library/a90k134d(v=VS.90).aspx


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