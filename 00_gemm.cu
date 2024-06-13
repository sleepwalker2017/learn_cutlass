#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>
#include <iostream>


int main() {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::RowMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::RowMajor,              // LayoutOutput
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // tag indicating Tensor Cores
    cutlass::arch::Sm80                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //
  int M = 16;
  int N = 16;
  int K = 16;

  float alpha = 1.f;
  float beta = 0.f;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> A({M, K});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> B({K, N});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> C({M, N});

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      A.host_ref().at({i, j}) = (i * K + j) % 5;
    } 
  }
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      B.host_ref().at({i, j}) = (i * N + j) % 5 + (i * N + j) % 4;
    } 
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C.host_ref().at({i, j}) = 0;
    } 
  }

  A.sync_device();
  B.sync_device();
  C.sync_device();

  cutlass::half_t const *ptrA = A.device_data();
  cutlass::half_t const *ptrB = B.device_data();
  cutlass::half_t const *ptrC = C.device_data();
  cutlass::half_t       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  C.sync_host();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C.host_ref().at({i, j}) << "\t";
    } 
    std::cout << std::endl;
  }

  return 0;
}
