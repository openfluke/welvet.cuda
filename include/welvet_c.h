#pragma once

/* Stable C ABI for welvet.cuda — used by w2a go_v_cuda and future NAC. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(WELVET_BUILD_SHARED)
#  ifdef WELVET_EXPORTS
#    define WELVET_API __declspec(dllexport)
#  else
#    define WELVET_API __declspec(dllimport)
#  endif
#else
#  define WELVET_API __attribute__((visibility("default")))
#endif

typedef struct welvet_status {
  int32_t code; /* 0=ok, 1=invalid, 2=unsupported, 3=gap, 4=internal */
  char message[256];
} welvet_status;

typedef struct welvet_dense_session welvet_dense_session;

WELVET_API const char* welvet_version(void);
WELVET_API const char* welvet_runtime_identity(void);

WELVET_API welvet_status welvet_cuda_init(void);
WELVET_API int32_t welvet_cuda_available(void);
WELVET_API int32_t welvet_gaps_count(void);
WELVET_API int32_t welvet_gaps_done(void);

/* Dense FormatNone forward — mount+execute+release. */
WELVET_API welvet_status welvet_dense_forward_f32(
    int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype, int32_t backend,
    const float* W_init, const double* bias, size_t bias_len, const float* x, float* pre,
    float* post);

/* Resident session for WVB2 bench: mount once, execute many. */
WELVET_API welvet_status welvet_dense_mount(int32_t batch, int32_t in, int32_t out, int32_t act,
                                            int32_t dtype, const float* W_init, const double* bias,
                                            size_t bias_len, welvet_dense_session** sess_out);
WELVET_API welvet_status welvet_dense_execute(welvet_dense_session* sess, const float* x, float* pre,
                                              float* post);
WELVET_API void welvet_dense_release(welvet_dense_session* sess);

#ifdef __cplusplus
}
#endif
