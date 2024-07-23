#include "bgmv_impl.cuh"
#include <vector>

using namespace std;

void print_vec(vector<half>& a, int col) {
	for(int i = 0; i < a.size(); i++) {
		std::cout << float(a[i]) << " ";
		if((i+1)%col == 0) {
			cout << endl;
		}
	}
}
vector<half> transpose(vector<half>& a, int m, int n) {
	vector<half> c(a);

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			int dst_idx = i * m + j;
			int src_idx = j * n + i;
			c[dst_idx] = a[src_idx];
		}
	}
	return c;
}

template<int m, int n, int k>
vector<half> compute_gpu(vector<half> &a, vector<half> &tmp_b) {
	auto b = transpose(tmp_b, k, n);
	half* d_a, *d_b;
	cudaMalloc(&d_a, a.size() * sizeof(half));
	cudaMalloc(&d_b, b.size() * sizeof(half));
	cudaMemcpy(d_a, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice);

	half* d_c;
	cudaMalloc(&d_c, m*n*sizeof(half));
	cudaMemset(d_c, m*n*sizeof(half), 0);

	int64_t * indices;
	cudaMalloc(&indices, m*sizeof(int64_t));
	cudaMemset(indices, m*sizeof(int64_t), 0);

	bgmv_kernel<k, n>(d_c, d_a, d_b, indices, m, 1, 0, 1);
	vector<half> c(m*n);
	cudaMemcpy(c.data(), d_c, m*n*sizeof(half), cudaMemcpyDeviceToHost);
	return c;
}
void test() {
	constexpr int batch = 2048;
	constexpr int r = 5120;
	constexpr int h2 = 32;

	vector<half> a(batch * r);
	vector<half> b(r * h2);

	for(int i = 0; i < a.size(); i++) {
		a[i] = i + 1;
	}
	for(int i = 0; i < b.size(); i++) {
		b[i] = i + 1;
	}

	//print_vec(b, h2);

	vector<half> right_c(batch * h2);
	for(int i = 0; i < batch; i++) {
		for(int j = 0; j < h2; j++) {
			double tmp = 0;
			for(int k = 0; k < r; k++) {
				auto x = (float(a[i* r + k]) * float(b[k*h2 + j]));
				//printf("a is %f, b is %f, x is %f\n", float(a[i* r + k]), float(b[k*h2 + j]), x);
				tmp += x;
			}
			right_c[i*h2 + j] = tmp;
		}
	}
	//print_vec(right_c, h2);

	auto c = compute_gpu<batch, h2, r>(std::ref(a), std::ref(b));
	//print_vec(c, h2);
#if 1
	for(int i = 0; i < c.size(); i++) {
		float tmp = std::abs(float(c[i]) - float(right_c[i]));
		if(tmp > 1e-6) {
			printf("error!!! %f vs %f\n", float(c[i]), float(right_c[i]));
			exit(0);
//exit(0);
		}
	}
	printf("all the same\n");
#endif
}

int main() {
	test();
	return 0;
}
