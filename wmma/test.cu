#include <vector>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda; 

#if 0
__global__ void wmma_ker(half *a, half *b, float *c) {
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; 
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; 
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
   wmma::fill_fragment(c_frag, 0.0f);
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

#else
__global__ void wmma_ker(half *a, half *b, float *c) {

   __shared__ half sha[16*16];
   __shared__ half shb[16*16];
   __shared__ float shc[16*16];

   int tid = threadIdx.x;

   for(int i = tid; i<256; i+=32) {
   	sha[i] = a[i];
	shb[i] = b[i];
   }
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; 
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; 
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag; 

   wmma::fill_fragment(c_frag, 0.0f);

   wmma::load_matrix_sync(a_frag, sha, 16);
   wmma::load_matrix_sync(b_frag, shb, 16);

   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   wmma::store_matrix_sync(shc, c_frag, 16, wmma::mem_row_major);
   for(int i = tid; i<256; i+=32) {
	c[i] = shc[i];
   }
}
#endif

// A : m * k
// B : k * n
// C : m * n

#if 0
template<typename T>
void gemm_cpu(T* a, T* b, T* c, int m, int n, int kk) {
    for(int i = 0; i < m; i++) {
	    for(int j = 0; j < n; j++) {
	    	for(int k = 0; k < kk; k++) {
			int offset_a = i * kk + k; // i , k
			int offset_b = k * n + j;  // k, j 
			int offset_c = i * n + j;  // i, j
			c[offset_c] += a[offset_a] * b[offset_b];
		}
	    }
    }
}

template<typename T>
void transpose(T* data, int m, int n) {
	std::vector<T> tmp(m * n);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			tmp[i * n + j] = data[i * n + j];
		}
	}
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			data[i * m + j] = tmp[j*n + i];
		}
	}
}

void test_transpose(int m, int k) {
	std::vector<float> a;

	a.resize(m * k);

	printf("A:\n");
	for(int i = 0; i < m * k; i++) {
		a[i] = i;
		printf("%.2f\t", a[i]);
		if((i+1)%k == 0) printf("\n");
	}
	
	transpose(a.data(), m, k);
	printf("after transpose:\n");
	for(int i = 0; i < m * k; i++) {
		printf("%.2f\t", a[i]);
		if((i+1)%k == 0) printf("\n");
	}

}
#endif

void test_gpu(int m, int n, int k) {
	std::vector<__half> a, b;

	a.resize(m * k);
	b.resize(n * k);

	std::vector<float> c;
	c.resize(m * n, 0);

	printf("A:\n");
	for(int i = 0; i < m * k; i++) {
		a[i] = i % 5;
		printf("%.2f\t", (float)a[i]);
		if((i+1)%n == 0) printf("\n");
	}

	printf("B:\n");
	for(int i = 0; i < n * k; i++) {
		b[i] = i % 5 + i % 4;
		printf("%.2f\t", (float)b[i]);
		if((i+1)%n == 0) printf("\n");
	}

	__half * d_a, *d_b;
	float *d_c;
	cudaMalloc(&d_a, sizeof(__half) * m * k);
	cudaMalloc(&d_b, sizeof(__half) * n * k);
	cudaMalloc(&d_c, sizeof(float) * m * n);

	//transpose(a.data(), m, k);
	//transpose(b.data(), k, n);
	cudaMemcpy(d_a, a.data(), sizeof(__half) * m * k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), sizeof(__half) * n * k, cudaMemcpyHostToDevice);
	//gemm_cpu(a.data(), b.data(), c.data(), m, n, k);
	wmma_ker<<<1, 32>>>(d_a, d_b, d_c);

	cudaMemcpy(c.data(), d_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

	printf("C output:\n");
	for(int i = 0; i < m * n; i++) {
		printf("%.2f\t", (float)c[i]);
		if((i+1)%n == 0) printf("\n");
	}
	printf("\n");
}

int main() {
	test_gpu(16, 16, 16);
	//test_transpose(16, 16);
	return 0;
}
