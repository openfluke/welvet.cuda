// Welvet CUDA kernels — M0 scaffold reference copy.
// The build compiles src/cuda/kernels.cu into the library.
// Dense MatVec / activations / backward kernels land as gates flip in GAPS.md.

// Identity saxpy-style smoke kernel: y[i] = a * x[i] + b
// Placeholder Dense GEMV (row-major W[out][in]): not wired from host Dense yet.
//
// See src/cuda/kernels.cu for the compiled definitions.
