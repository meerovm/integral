
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <string>
#include <omp.h>
#include <vector>

#if (defined(__riscv))
#include "openrand/philox.h"
#include <riscv_vector.h>
#endif

#include <chrono>
#include <thread>

#include <cstdlib>
#include <iomanip>
#include <fstream>
//#include <bit>

#if (!defined(__riscv))
#include <mkl.h>
#include <mkl_vsl.h>
#endif

#include <cstdint>
#include <cfloat>
#include <cmath>

#if (defined(__riscv))
class sobol
{
private:

    const unsigned d;
    unsigned* x;
    unsigned Ns = 1;
    const unsigned max_length;
    unsigned** V;
    unsigned bufer[460] = { 2, 1, 0, 1, 3, 2, 1, 1, 3, 4, 3, 1, 1, 3, 1, 5, 3, 2, 1, 1, 1, 6, 4, 1, 1, 1, 3, 3, 7, 4, 4, 1, 3, 5, 13, 8, 5, 2, 1, 1, 5, 5, 17, 9, 5, 4, 1, 1, 5, 5, 5, 10, 5, 7, 1, 1, 7, 11, 19, 11, 5, 11, 1, 1, 5, 1, 1, 12, 5, 13, 1, 1, 1, 3, 11, 13, 5, 14, 1, 3, 5, 5, 31, 14, 6, 1, 1, 3, 3, 9, 7, 49, 15, 6, 13, 1, 1, 1, 15, 21, 21, 16, 6, 16, 1, 3, 1, 13, 27, 49, 17, 6, 19, 1, 1, 1, 15, 7, 5, 18, 6, 22, 1, 3, 1, 15, 13, 25, 19, 6, 25, 1, 1, 5, 5, 19, 61, 20, 7, 1, 1, 3, 7, 11, 23, 15, 103, 21, 7, 4, 1, 3, 7, 13, 13, 15, 69, 22, 7, 7, 1, 1, 3, 13, 7, 35, 63, 23, 7, 8, 1, 3, 5, 9, 1, 25, 53, 24, 7, 14, 1, 3, 1, 13, 9, 35, 107, 25, 7, 19, 1, 3, 1, 5, 27, 61, 31, 26, 7, 21, 1, 1, 5, 11, 19, 41, 61, 27, 7, 28, 1, 3, 5, 3, 3, 13, 69, 28, 7, 31, 1, 1, 7, 13, 1, 19, 1, 29, 7, 32, 1, 3, 7, 5, 13, 19, 59, 30, 7, 37, 1, 1, 3, 9, 25, 29, 41, 31, 7, 41, 1, 3, 5, 13, 23, 1, 55, 32, 7, 42, 1, 3, 7, 3, 13, 59, 17, 33, 7, 50, 1, 3, 1, 3, 5, 53, 69, 34, 7, 55, 1, 1, 5, 5, 23, 33, 13, 35, 7, 56, 1, 1, 7, 7, 1, 61, 123, 36, 7, 59, 1, 1, 7, 9, 13, 61, 49, 37, 7, 62, 1, 3, 3, 5, 3, 55, 33, 38, 8, 14, 1, 3, 1, 15, 31, 13, 49, 245, 39, 8, 21, 1, 3, 5, 15, 31, 59, 63, 97, 40, 8, 22, 1, 3, 1, 11, 11, 11, 77, 249, 41, 8, 38, 1, 3, 1, 11, 27, 43, 71, 9, 42, 8, 47, 1, 1, 7, 15, 21, 11, 81, 45, 43, 8, 49, 1, 3, 7, 3, 25, 31, 65, 79, 44, 8, 50, 1, 3, 1, 1, 19, 11, 3, 205, 45, 8, 52, 1, 1, 5, 9, 19, 21, 29, 157, 46, 8, 56, 1, 3, 7, 11, 1, 33, 89, 185, 47, 8, 67, 1, 3, 3, 3, 15, 9, 79, 71, 48, 8, 70, 1, 3, 7, 11, 15, 39, 119, 27, 49, 8, 84, 1, 1, 3, 1, 11, 31, 97, 225, 50, 8, 97, 1, 1, 1, 3, 23, 43, 57, 177 };

public:

    // Конструктор принимает размерность dm <= 50, максимальное кол-во точек, которое может быть сгенерированно ml, файл dir_file
    // Требует O(dm * log(ml)) памяти

    sobol(const unsigned dm, const unsigned ml) : max_length(ml), d(dm)
    {
        if (dm > 50) {
            std::cout << "size d exceeded\n";
            exit(1);
        }
        int bufer_index = 0;
        const unsigned L = (unsigned)ceil(log((double)(ml)) / log(2.0));
        V = new unsigned* [d];
        V[0] = new unsigned[L + 1];
        for (unsigned i = 1; i <= L; i++) V[0][i] = 1 << (32 - i);
        for (int j = 1; j <= d - 1; j++) {
            V[j] = new unsigned[L + 1];
            unsigned d, s, a;
            d = bufer[bufer_index];
            s = bufer[bufer_index + 1];
            a = bufer[bufer_index + 2];
            bufer_index += 3;
            if (L <= s) {
                const int start = bufer_index - 1;
                for (; bufer_index - start <= L; bufer_index++) V[j][bufer_index - start] = bufer[bufer_index] << (32 - bufer_index + start);
            }
            else {
                const int start = bufer_index - 1;
                for (; bufer_index - start <= s; bufer_index++) V[j][bufer_index - start] = bufer[bufer_index] << (32 - bufer_index + start);
                for (unsigned i = s + 1; i <= L; i++) {
                    V[j][i] = V[j][i - s] ^ (V[j][i - s] >> s);
                    for (unsigned k = 1; k <= s - 1; k++) {
                        V[j][i] ^= (((a >> (s - 1 - k)) & 1) * V[j][i - k]);
                    }
                }
            }
        }
        x = new unsigned[d];
        for (int i = 0; i < d; i++) { x[i] = 0; }
    }


    ~sobol() {
        delete[]x;
        for (int i = 0; i < d; i++) {
            delete[] V[i];
        }
        delete[] V;
    }

    // Генерируем N точек начиная со следующей, после последней сгенерированной, D - размерность

    double* sobol_points(double* POINTS, const unsigned N, const unsigned D)
    {
        if ((Ns + N - 1) > max_length) {
            std::cout << "the count of numbers more the maximum";
            exit(1);
        }

        double p = pow(2.0, 32);
        for (unsigned j = 0; j <= D - 1; j++) POINTS[j] = 0;
        POINTS[0] = x[0];
        for (unsigned i = 0; i <= N - 2; i++) {
            POINTS[(i + 1) * D] = (unsigned)POINTS[i * D] ^ (V[0][std::countr_one(i + Ns - 1) + 1]);
            POINTS[i * D] /= p;
        }
        x[0] = (unsigned)POINTS[(N - 1) * D] ^ (V[0][std::countr_one(N - 2 + Ns) + 1]);
        POINTS[(N - 1) * D] /= p;
        for (unsigned j = 1; j <= D - 1; j++) {
            POINTS[j] = x[j];
            for (unsigned i = 0; i <= N - 2; i++) {
                POINTS[(i + 1) * D + j] = (unsigned)POINTS[i * D + j] ^ (V[j][std::countr_one(i + Ns - 1) + 1]);
                POINTS[i * D + j] /= p;
            }
            x[j] = (unsigned)POINTS[(N - 1) * D + j] ^ (V[j][std::countr_one(N - 2 + Ns) + 1]);
            POINTS[(N - 1) * D + j] /= p;
        }
        Ns += N;
        return POINTS;
    }

    // пропускает первые k точек с начала (не с текущей точки)
    // работает за O(d)
    void skip_ahead(const unsigned k)
    {
        Ns = k;
        unsigned g;
        for (unsigned j = 0; j < d; j++) {
            unsigned xx = 0;
            g = Ns ^ (Ns >> 1);
            for (unsigned l = 0; l < 32; l++) {
                if ((g >> l) & 1) {
                    xx ^= V[j][l + 1];
                }
            }
            x[j] = xx;
        }
        Ns++;
    }
};
#endif

#if (defined(__riscv))
#include"exp.cpp"
#endif

template <typename T, typename Q>
class montecarlo {
private:
    T f;
#if (defined(__riscv))
    Q f_vector;
#endif
public:
#if (defined(__riscv))
    montecarlo(T ff, Q qq) : f(ff), f_vector(qq) {}
#else
    montecarlo(T ff) : f(ff) {}
#endif

    double left_rectangle(double l, double r, long long int count_point) {
        const double step = (r - l) / count_point;
        double ans = 0;
        for (double point = l; point < r; point += step) {
            ans += f(point);
        }
        return ans * step;
    }

    double left_rectangle_parallel(double l, double r, long long int count_point, int threads) {
        const double step = (r - l) / count_point;
        double ans = 0;
        double point;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(point)
        for (int i = 0; i < count_point; i++) {
            point = l + i * step;
            ans += f(point);
        }
        return ans * step;
    }

    double right_rectangle(double l, double r, long long int count_point) {
        const double step = (r - l) / count_point;
        double ans = 0;
        for (double point = l + step; point <= r; point += step) {
            ans += f(point);
        }
        return ans * step;
    }

    double right_rectangle_parallel(double l, double r, long long int count_point, int threads) {
        const double step = (r - l) / count_point;
        double ans = 0;
        double point;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(point)
        for (int i = 1; i < count_point + 1; i++) {
            point = l + i * step;
            ans += f(point);
        }
        return ans * step;
    }

    double sentr_rectangle(double l, double r, long long int count_point) {
        const double step = (r - l) / count_point;
        double ans = 0;
        for (double point = l + step / 2; point < r; point += step) {
            ans += f(point);
        }
        return ans * step;
    }

    double sentr_rectangle_parallel(double l, double r, long long int count_point, int threads) {
        const double step = (r - l) / count_point;
        double ans = 0;
        double point;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(point)
        for (int i = 0; i < count_point; i++) {
            point = l + (i + 0.5) * step;
            ans += f(point);
        }
        return ans * step;
    }

#if (defined(__riscv))
    double sentr_rectangle_parallel_vector(double l, double r, long long int count_point, int threads) {
        const double step = (r - l) / count_point;
        omp_set_num_threads(threads);
        double ans = 0;
        size_t vl = __riscv_vsetvlmax_e64m2();
#pragma omp parallel reduction(+ : ans)
        {
            int t = omp_get_thread_num();
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step * threads, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l + step / 2 + t * step * vl, vl), step, v_idx_d, vl);
            for (long long int i = t; i < count_point / vl; i += threads) {
                vfloat64m2_t v_ans_n = f_vector(v_point, vl);
                v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
            }

            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
            ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
        }
        if (count_point % vl != 0) {
            for (double point = r - (count_point) % vl * step + step / 2; point < r; point += step) {
                ans += f(point);
            }
        }
        return ans * step;
    }
#endif

    double trapezoid(double l, double r, long long int count_point) {
        const double step = (r - l) / count_point;
        double ans = (f(l) + f(r)) / 2;
        double point = step;
        for (long long int i = 1; i < count_point; i++) {
            ans += f(point);
            point += step;
        }
        return ans * step;
    }

    double trapezoid_parallel(double l, double r, long long int count_point, int threads) {
        const double step = (r - l) / count_point;
        double ans = (f(l) + f(r)) / 2;
        double point;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(point)
        for (long long int i = 1; i <= count_point - 1; i++) {
            point = l + (i)*step;
            ans += f(point);
        }
        return ans * step;
    }

#if (defined(__riscv))
    double trapezoid_parallel_vector(double l, double r, long long int count_point, int threads) {
        const double step = (r - l) / count_point;
        omp_set_num_threads(threads);
        double ans = 0;
        size_t vl = __riscv_vsetvlmax_e64m2();
#pragma omp parallel reduction(+ : ans)
        {
            int t = omp_get_thread_num();
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step * threads, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l + t * step * vl, vl), step, v_idx_d, vl);
            for (long long int i = t; i < count_point / vl; i += threads) {
                vfloat64m2_t v_ans_n = f_vector(v_point, vl);
                v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
            }

            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
            ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
        }

        if (count_point % vl != 0) {
            for (double point = r - (count_point) % vl * step; point <= r; point += step) {
                ans += f(point);
            }
        }
        ans -= (f(l) + f(r)) / 2;
        return ans * step;
    }

    double base_montekarlo(double l, double r, long long int count_point) {
        double ans = 0;
        openrand::Philox rng(1234, 0);
        for (int i = 0; i < count_point; i++) {
            ans += f(l + (rng.template rand<double>() * (r - l)));
        }
        return ans * (r - l) / count_point;
    }

    double base_montekarlo_parallel(double l, double r, long long int count_point, int threads) {
        double ans = 0;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            int th_n = omp_get_thread_num();
            openrand::Philox rng(1234, 0, th_n);
#pragma omp for
            for (long long int i = 0; i < count_point; i++) {
                ans += f(l + (rng.template rand<double>() * (r - l)));
            }
        }
        return ans * (r - l) / count_point;
    }

    double base_montekarlo_parallel_vector(double l, double r, long long int count_point, int threads) {
        omp_set_num_threads(threads);
        size_t vl = __riscv_vsetvlmax_e64m2();
        const int bufer_size = 1000 * vl;
        double ans = 0;
#pragma omp parallel reduction(+ : ans)
        {
            int th_n = omp_get_thread_num();
            openrand::Philox rng(1234, 0, th_n);
            double* bufer = new double[bufer_size];
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m2_t v_point = __riscv_vfmv_v_f_f64m2(0.0, vl);
#pragma omp for
            for (int i = 0; i < count_point / bufer_size; i++) {
                for (int j = 0; j < bufer_size; j++) {
                    bufer[j] = l + (rng.template rand<double>() * (r - l));
                }
                for (int j = 0; j < bufer_size; j += vl) {
                    v_point = __riscv_vle64_v_f64m2(bufer + j, vl);
                    vfloat64m2_t v_ans_n = f_vector(v_point, vl);
                    v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                }
            }
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
            ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
            delete[] bufer;
        }

        vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
        vfloat64m2_t v_point = __riscv_vfmv_v_f_f64m2(0.0, vl);
        int rem = (count_point % bufer_size) - (count_point % bufer_size) % vl;
        double* bufer = new double[rem];
        openrand::Philox rng(1234, 0, threads + 1);
        for (int j = 0; j < rem; j++) {
            bufer[j] = l + (rng.template rand<double>() * (r - l));
        }
        for (int j = 0; j < rem; j += vl) {
            v_point = __riscv_vle64_v_f64m2(bufer + j, vl);
            vfloat64m2_t v_ans_n = f_vector(v_point, vl);
            v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
        }
        delete[] bufer;
        for (int i = 0; i < (count_point % bufer_size) % vl; i++) {
            ans += f(l + (rng.template rand<double>() * (r - l)));
        }
        vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
        vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
        ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
        return ans * (r - l) / count_point;
    }

    double sobol_montekarlo(double l, double r, long long int count_point) {
        double ans = 0;
        const int bufer_size = 10000;
        sobol s(1, count_point);
        double* p = new double[bufer_size];
        for (int i = 0; i < count_point / bufer_size; i++) {
            s.sobol_points(p, bufer_size, 1);
            for (int j = 0; j < bufer_size; j++) {
                ans += f(l + p[j] * (r - l));
            }
        }
        delete[] p;

        if (count_point % bufer_size != 0) {
            double* q = new double[count_point % bufer_size];
            s.sobol_points(q, count_point % bufer_size, 1);
            for (int i = 0; i < count_point % bufer_size; i++) {
                ans += f(l + q[i] * (r - l));
            }
            delete[]q;
        }
        return ans * (r - l) / count_point;
    }

    double sobol_montekarlo_parallel(double l, double r, long long int count_point, int threads) {
        double ans = 0;
        const int bufer_size = 10000;
        int th = threads;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            int th_n = omp_get_thread_num();
            long long int st = th_n * (count_point / th);
            long long int fin = (th_n + 1) * (count_point / th);
            if (th_n == th - 1) {
                fin = count_point;
            }
            long long int c_p = fin - st;
            sobol sobol(1, count_point);
            sobol.skip_ahead(st);
            double* p = new double[bufer_size];
            for (int i = 0; i < c_p / bufer_size; i++) {
                sobol.sobol_points(p, bufer_size, 1);
                for (int j = 0; j < bufer_size; j++) {
                    ans += f(l + p[j] * (r - l));
                }
            }
            delete[] p;
            if (c_p % bufer_size != 0) {
                double* q = new double[c_p % bufer_size];
                sobol.sobol_points(q, c_p % bufer_size, 1);
                for (int i = 0; i < c_p % bufer_size; i++) {
                    ans += f(l + q[i] * (r - l));
                }
                delete[]q;
            }
        }
        return ans * (r - l) / count_point;
    }

    double sobol_montekarlo_parallel_vector(double l, double r, long long int count_point, int threads) {
        size_t vl = __riscv_vsetvlmax_e64m2();
        double ans = 0;
        const int bufer_size = 1000 * vl;
        int th = threads;
#pragma omp parallel reduction(+ : ans)
        {
            int th_n = omp_get_thread_num();
            long long int st = th_n * (count_point / th);
            long long int fin = (th_n + 1) * (count_point / th);
            if (th_n == th - 1) {
                fin = count_point;
            }
            long long int c_p = fin - st;
            sobol s(1, count_point);
            s.skip_ahead(st);
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m2_t v_point = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m2_t v_l = __riscv_vfmv_v_f_f64m2(l, vl);
            vfloat64m2_t v_r = __riscv_vfmv_v_f_f64m2(r - l, vl);
            double* p = new double[bufer_size];
            for (int i = 0; i < c_p / bufer_size; i++) {
                p = s.sobol_points(p, bufer_size, 1);
                for (int j = 0; j < bufer_size; j += vl) {
                    v_point = __riscv_vle64_v_f64m2(p + j, vl);
                    v_point = __riscv_vfmul_vv_f64m2(v_point, v_r, vl);
                    v_point = __riscv_vfadd_vv_f64m2(v_point, v_l, vl);
                    vfloat64m2_t v_ans_n = f_vector(v_point, vl);
                    v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                }
            }
            delete[] p;

            int rem = (c_p % bufer_size) - (c_p % bufer_size) % vl;
            if (rem != 0) {
                double* q = new double[rem];
                s.sobol_points(q, rem, 1);
                for (int j = 0; j < rem; j += vl) {
                    v_point = __riscv_vle64_v_f64m2(q + j, vl);
                    v_point = __riscv_vfmul_vv_f64m2(v_point, v_r, vl);
                    v_point = __riscv_vfadd_vv_f64m2(v_point, v_l, vl);
                    vfloat64m2_t v_ans_n = f_vector(v_point, vl);
                    v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                }
                delete[] q;
            }
            if ((c_p % bufer_size) % vl != 0) {
                double* w = new double[(c_p % bufer_size) % vl];
                s.sobol_points(w, (c_p % bufer_size) % vl, 1);
                for (int i = 0; i < (c_p % bufer_size) % vl; i++) {
                    ans += f(l + w[i] * (r - l));
                }
                delete[] w;
            }
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
            ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
        }
        return ans * (r - l) / count_point;
    }

#else

    double base_montekarlo(double l, double r, long long int count_point) {
        double ans = 0;
        const int bufer_sz = 1000;
        double p[1000];
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, 777);
        for (int i = 0; i < count_point / bufer_sz; i++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, bufer_sz, p, l, r);
            for (int j = 0; j < bufer_sz; j++) {
                ans += f(p[j]);
            }
        }
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, count_point % bufer_sz, p, l, r);
        for (int j = 0; j < count_point % bufer_sz; j++) {
            ans += f(p[j]);
        }
        vslDeleteStream(&stream);
        return ans * (r - l) / count_point;
    }

    double base_montekarlo_parallel(double l, double r, long long int count_point, int threads) {
        const int bufer_sz = 1000;
        double ans = 0;
        double k = (r - l);
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, 777);
        const int th = threads;
        VSLStreamStatePtr* streams = new VSLStreamStatePtr[th];
        for (int i = 0; i < th; i++) {
            vslCopyStream(&streams[i], stream);
            vslSkipAheadStream(streams[i], i * (count_point / th));
        }
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            int t = omp_get_thread_num();
            double p[1000];
            for (int i = 0; i < count_point / th / bufer_sz; i++) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], bufer_sz, p, l, r);
                for (int j = 0; j < bufer_sz; j++) {
                    ans += f(p[j]);
                }
            }
            const int rem = (count_point / th) % bufer_sz;
            if (rem != 0) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], rem, p, l, r);
                for (int j = 0; j < rem; j++) {
                    ans += f(p[j]);
                }
            }
        }
        for (int i = 0; i < th; i++) {
            vslDeleteStream(&streams[i]);
        }
        delete[] streams;
        vslDeleteStream(&stream);
        return ans * k / count_point;
    }

    double sobol_montekarlo(double l, double r, long long int count_point) {
        double ans = 0;
        const int bufer_sz = 1000;
        double p[1000];
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SOBOL, 777);
        for (int i = 0; i < count_point / bufer_sz; i++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, bufer_sz, p, l, r);
            for (int j = 0; j < bufer_sz; j++) {
                ans += f(p[j]);
            }
        }
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, count_point % bufer_sz, p, l, r);
        for (int j = 0; j < count_point % bufer_sz; j++) {
            ans += f(p[j]);
        }
        vslDeleteStream(&stream);
        return ans * (r - l) / count_point;
    }

    double sobol_montekarlo_parallel(double l, double r, long long int count_point, int threads) {
        const int bufer_sz = 1000;
        double ans = 0;
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SOBOL, 1);
        const int th = threads;
        VSLStreamStatePtr* streams = new VSLStreamStatePtr[th];
        for (int i = 0; i < th; i++) {
            vslCopyStream(&streams[i], stream);
            vslSkipAheadStream(streams[i], i * (count_point / th));
        }
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            int t = omp_get_thread_num();
            double p[1000];
            for (int i = 0; i < count_point / th / bufer_sz; i++) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], bufer_sz, p, l, r);
                for (int j = 0; j < bufer_sz; j++) {
                    ans += f(p[j]);
                }
            }
            const int rem = (count_point / th) % bufer_sz;
            if (rem != 0) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], rem, p, l, r);
                for (int j = 0; j < rem; j++) {
                    ans += f(p[j]);
                }
            }

        }
        for (int i = 0; i < th; i++) {
            vslDeleteStream(&streams[i]);
        }
        delete[] streams;
        vslDeleteStream(&stream);
        return ans * (r - l) / count_point;
    }


#endif

    double sentr_rectangle_2(double l_x, double r_x, double l_y, double r_y, int c_x, int c_y) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double ans = 0, p_y = 0;
        double p_x = l_x + step_x / 2;
        for (int i = 0; i < c_x; i++) {
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                ans += f(p_x, p_y);
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y;
    }

    double sentr_rectangle_2_parallel(double l_x, double r_x, double l_y, double r_y, int c_x, int c_y, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double ans = 0;
        double p_x = l_x + step_x / 2;
        double p_y = l_y + step_y / 2;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y)
        for (int i = 0; i < c_x; i++) {
            p_x = l_x + step_x * (i + 0.5);
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                ans += f(p_x, p_y);
                p_y += step_y;
            }
        }
        return ans * step_x * step_y;
    }

#if (defined(__riscv))
    double sentr_rectangle_2_parallel_vector(double l_x, double r_x, double l_y, double r_y, int c_x, int c_y, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;

        omp_set_num_threads(threads);
        double ans = 0;
        size_t vl = __riscv_vsetvlmax_e64m2();
#pragma omp parallel reduction(+ : ans)
        {
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_y, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
#pragma omp for
            for (int i = 0; i < c_x; i++) {
                double p_x = l_x + step_x / 2.0 + i * step_x;
                vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
                vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_y + step_y / 2, vl), step_y, v_idx_d, vl);
                for (long long int j = 0; j < c_y / vl; j++) {
                    vfloat64m2_t v_ans_n = f_vector(p_x, v_point, vl);
                    v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                    v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                }

                vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
                ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
                if (c_y % vl != 0) {
                    for (double point = r_y - (c_y) % vl * step_y + step_y / 2; point < r_y; point += step_y) {
                        ans += f(p_x, point);
                    }
                }
            }
        }
        return ans * step_x * step_y;
    }
#endif

    double trapezoid_2(double l_x, double r_x, double l_y, double r_y, int c_x, int c_y) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double ans = 0, p_y = 0, ans_y;
        double p_x = l_x;
        double k_x;
        for (int i = 0; i <= c_x; i++) {
            p_y = l_y;
            k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
            ans_y = 0.5 * f(p_x, p_y);
            p_y += step_y;
            for (int j = 1; j < c_y; j++) {
                ans_y += f(p_x, p_y);
                p_y += step_y;
            }
            ans_y += 0.5 * f(p_x, p_y);
            ans += k_x * ans_y;
            p_x += step_x;
        }
        return ans * step_x * step_y;
    }

    double trapezoid_2_parallel(double l_x, double r_x, double l_y, double r_y, int c_x, int c_y, int threads) {
        omp_set_num_threads(threads);
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double ans = 0;
#pragma omp parallel reduction(+ : ans)
        {
            double p_x = l_x, ans_y, k_x, p_y;
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                ans_y = 0.5 * f(p_x, p_y);
                p_y += step_y;
                for (int j = 1; j < c_y; j++) {
                    ans_y += f(p_x, p_y);
                    p_y += step_y;
                }
                ans_y += 0.5 * f(p_x, p_y);
                ans += k_x * ans_y;
            }
        }
        return ans * step_x * step_y;
    }

#if (defined(__riscv))
    double trapezoid_2_parallel_vector(double l_x, double r_x, double l_y, double r_y, int c_x, int c_y, int threads) {
        size_t vl = __riscv_vsetvlmax_e64m2();
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double ans = 0;
        c_y--;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            double k_x, ans_y, p_x;
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_y, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_y + step_y, vl), step_y, v_idx_d, vl);
                ans_y = 0.5 * f(p_x, l_y);
                vfloat64m2_t v_ans_y = __riscv_vfmv_v_f_f64m2(0.0, vl);
                for (long long int k = 0; k < c_y / vl; k++) {
                    vfloat64m2_t v_ans_n = f_vector(p_x, v_point, vl);
                    v_ans_y = __riscv_vfadd_vv_f64m2(v_ans_y, v_ans_n, vl);
                    v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                }
                vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans_y, v_zero, vl);
                ans_y += __riscv_vfmv_f_s_f64m1_f64(v_sum);
                if (c_y % vl != 0) {
                    for (double point = r_y - (c_y) % vl * step_y; point < r_y; point += step_y) {
                        ans_y += f(p_x, point);
                    }
                }
                ans_y += 0.5 * f(p_x, r_y);
                ans += k_x * ans_y;
            }
        }
        return ans * step_x * step_y;
    }
#endif

    double sentr_rectangle_3(double l_x, double r_x, double l_y, double r_y,
        double l_z, double r_z, int c_x, int c_y, int c_z) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double ans = 0, p_y = 0, p_z = 0;
        double p_x = l_x + step_x / 2;
        for (int i = 0; i < c_x; i++) {
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                p_z = l_z + step_z / 2;
                for (int k = 0; k < c_z; k++) {
                    ans += f(p_x, p_y, p_z);
                    p_z += step_z;
                }
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y * step_z;
    }

    double sentr_rectangle_3_parallel(double l_x, double r_x, double l_y, double r_y,
        double l_z, double r_z, int c_x, int c_y, int c_z, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double ans = 0;
        double p_x = l_x + step_x / 2;
        double p_y = l_y + step_y / 2;
        double p_z = l_z + step_z / 2;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z)
        for (int i = 0; i < c_x; i++) {
            p_x = l_x + (i + 0.5) * step_x;
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                p_z = l_z + step_z / 2;
                for (int k = 0; k < c_z; k++) {
                    ans += f(p_x, p_y, p_z);
                    p_z += step_z;
                }
                p_y += step_y;
            }
        }
        return ans * step_x * step_y * step_z;
    }

#if (defined(__riscv))
    double sentr_rectangle_3_parallel_vector(double l_x, double r_x, double l_y, double r_y,
        double l_z, double r_z, int c_x, int c_y, int c_z, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        omp_set_num_threads(threads);
        double ans = 0;
        size_t vl = __riscv_vsetvlmax_e64m2();
#pragma omp parallel reduction(+ : ans)
        {
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_y, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
#pragma omp for
            for (int i = 0; i < c_x; i++) {
                v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
                double p_x = l_x + step_x / 2.0 + i * step_x;
                for (int k = 0; k < c_z; k++) {
                    double p_z = l_z + step_z / 2.0 + k * step_z;
                    vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_y + step_y / 2, vl), step_y, v_idx_d, vl);
                    for (long long int j = 0; j < c_y / vl; j++) {
                        vfloat64m2_t v_ans_n = f_vector(p_x, v_point, p_z, vl);
                        v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                        v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                    }
                    if (c_y % vl != 0) {
                        for (double point = r_y - (c_y) % vl * step_y + step_y / 2; point < r_y; point += step_y) {
                            ans += f(p_x, point, p_z);
                        }
                    }
                }
                vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
                ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
            }
        }
        return ans * step_x * step_y * step_z;
    }
#endif

    double trapezoid_3(double l_x, double r_x, double l_y, double r_y,
        double l_z, double r_z, int c_x, int c_y, int c_z) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double ans = 0;
        double p_x = l_x;
        double p_y = l_y;
        double p_z = l_z;
        double k_x, k_y, ans_z;
        for (int i = 0; i <= c_x; i++) {
            p_y = l_y;
            k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
            for (int j = 0; j <= c_y; j++) {
                p_z = l_z;
                k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                ans_z = 0.5 * f(p_x, p_y, p_z);
                p_z += step_z;
                for (int k = 1; k < c_z; k++) {
                    ans_z += f(p_x, p_y, p_z);
                    p_z += step_z;
                }
                ans_z += 0.5 * f(p_x, p_y, p_z);
                ans += k_x * k_y * ans_z;
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y * step_z;
    }

    double trapezoid_3_parallel(double l_x, double r_x, double l_y, double r_y,
        double l_z, double r_z, int c_x, int c_y, int c_z, int threads) {
        omp_set_num_threads(threads);
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double ans = 0;
#pragma omp parallel reduction(+ : ans)
        {
            double p_x, p_y, p_z, k_x, k_y, ans_z;
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                for (int j = 0; j <= c_y; j++) {
                    p_z = l_z;
                    k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                    ans_z = 0.5 * f(p_x, p_y, p_z);
                    p_z += step_z;
                    for (int k = 1; k < c_z; k++) {
                        ans_z += f(p_x, p_y, p_z);
                        p_z += step_z;
                    }
                    ans_z += 0.5 * f(p_x, p_y, p_z);
                    ans += k_x * k_y * ans_z;
                    p_y += step_y;
                }
            }
        }
        return ans * step_x * step_y * step_z;
    }

#if (defined(__riscv))
    double trapezoid_3_parallel_vector(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, int c_x, int c_y, int c_z, int threads) {
        size_t vl = __riscv_vsetvlmax_e64m2();
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double ans = 0;
        c_z--;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            double k_x, k_y, ans_z, p_x, p_y;
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_z, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                for (int j = 0; j <= c_y; j++) {
                    vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_z + step_z, vl), step_z, v_idx_d, vl);
                    k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                    ans_z = 0.5 * f(p_x, p_y, l_z);
                    vfloat64m2_t v_ans_z = __riscv_vfmv_v_f_f64m2(0.0, vl);
                    for (long long int k = 0; k < c_z / vl; k++) {
                        vfloat64m2_t v_ans_n = f_vector(p_x, p_y, v_point, vl);
                        v_ans_z = __riscv_vfadd_vv_f64m2(v_ans_z, v_ans_n, vl);
                        v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                    }
                    vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans_z, v_zero, vl);
                    ans_z += __riscv_vfmv_f_s_f64m1_f64(v_sum);
                    if (c_z % vl != 0) {
                        for (double point = r_z - (c_z) % vl * step_z; point < r_z; point += step_z) {
                            ans_z += f(p_x, p_y, point);
                        }
                    }
                    ans_z += 0.5 * f(p_x, p_y, r_z);
                    ans += k_x * k_y * ans_z;
                    p_y += step_y;
                }
            }
        }
        return ans * step_x * step_y * step_z;
    }
#endif

    double sentr_rectangle_4(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double ans = 0, p_y = 0, p_z = 0, p_w = 0;
        double p_x = l_x + step_x / 2;
        for (int i = 0; i < c_x; i++) {
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                p_z = l_z + step_z / 2;
                for (int k = 0; k < c_z; k++) {
                    p_w = l_w + step_w / 2;
                    for (int e = 0; e < c_w; e++) {
                        ans += f(p_x, p_y, p_z, p_w);
                        p_w += step_w;
                    }
                    p_z += step_z;
                }
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y * step_z * step_w;
    }

    double sentr_rectangle_4_parallel(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double ans = 0;
        double p_x = l_x + step_x / 2;
        double p_y = l_y + step_y / 2;
        double p_z = l_z + step_z / 2;
        double p_w = l_w + step_w / 2;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z, p_w)
        for (int i = 0; i < c_x; i++) {
            p_x = l_x + (i + 0.5) * step_x;
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                p_z = l_z + step_z / 2;
                for (int k = 0; k < c_z; k++) {
                    p_w = l_w + step_w / 2;
                    for (int e = 0; e < c_w; e++) {
                        ans += f(p_x, p_y, p_z, p_w);
                        p_w += step_w;
                    }
                    p_z += step_z;
                }
                p_y += step_y;
            }
        }
        return ans * step_x * step_y * step_z * step_w;
    }

#if (defined(__riscv))
    double sentr_rectangle_4_parallel_vector(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        omp_set_num_threads(threads);
        double ans = 0;
        size_t vl = __riscv_vsetvlmax_e64m2();
#pragma omp parallel reduction(+ : ans)
        {
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_y, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
#pragma omp for
            for (int i = 0; i < c_x; i++) {
                double p_x = l_x + step_x / 2.0 + i * step_x;
                v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
                for (int e = 0; e < c_w; e++) {
                    double p_w = l_w + step_w / 2.0 + e * step_w;
                    for (int k = 0; k < c_z; k++) {
                        double p_z = l_z + step_z / 2.0 + k * step_z;
                        vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_y + step_y / 2, vl), step_y, v_idx_d, vl);
                        for (long long int j = 0; j < c_y / vl; j++) {
                            vfloat64m2_t v_ans_n = f_vector(p_x, v_point, p_z, p_w, vl);
                            v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                            v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                        }
                        if (c_y % vl != 0) {
                            for (double point = r_y - (c_y) % vl * step_y + step_y / 2; point < r_y; point += step_y) {
                                ans += f(p_x, point, p_z, p_w);
                            }
                        }
                    }
                }
                vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
                ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
            }
        }
        return ans * step_x * step_y * step_z * step_w;
    }
#endif

    double trapezoid_4(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double ans = 0;
        double p_x = l_x;
        double p_y = l_y;
        double p_z = l_z;
        double p_w = l_w;
        double k_x, k_y, k_z, ans_w;
        for (int i = 0; i <= c_x; i++) {
            p_y = l_y;
            k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
            for (int j = 0; j <= c_y; j++) {
                p_z = l_z;
                k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                for (int k = 0; k <= c_z; k++) {
                    p_w = l_w;
                    k_z = 1 - 0.5 * ((k == 0) | (k == c_z));
                    ans_w = 0.5 * f(p_x, p_y, p_z, p_w);
                    p_w += step_w;
                    for (int e = 1; e < c_w; e++) {
                        ans_w += f(p_x, p_y, p_z, p_w);
                        p_w += step_w;
                    }
                    ans_w += 0.5 * f(p_x, p_y, p_z, p_w);
                    ans += k_x * k_y * k_z * ans_w;
                    p_z += step_z;
                }
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y * step_z * step_w;
    }

    double trapezoid_4_parallel(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w, int threads) {
        omp_set_num_threads(threads);
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double ans = 0;
#pragma omp parallel reduction(+ : ans)
        {
            double p_x, p_y, p_z, p_w, k_x, k_y, k_z, ans_w;
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                for (int j = 0; j <= c_y; j++) {
                    p_z = l_z;
                    k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                    for (int k = 0; k <= c_z; k++) {
                        p_w = l_w;
                        k_z = 1 - 0.5 * ((k == 0) | (k == c_z));
                        ans_w = 0.5 * f(p_x, p_y, p_z, p_w);
                        p_w += step_w;
                        for (int e = 1; e < c_w; e++) {
                            ans_w += f(p_x, p_y, p_z, p_w);
                            p_w += step_w;
                        }
                        ans_w += 0.5 * f(p_x, p_y, p_z, p_w);
                        ans += k_x * k_y * k_z * ans_w;
                        p_z += step_z;
                    }
                    p_y += step_y;
                }
            }
        }
        return ans * step_x * step_y * step_z * step_w;
    }

#if (defined(__riscv))
    double trapezoid_4_parallel_vector(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w, int c_x, int c_y, int c_z, int c_w, int threads) {
        size_t vl = __riscv_vsetvlmax_e64m2();
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;

        double ans = 0;
        c_w--;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            double k_x, k_y, k_z, ans_w, p_x, p_y, p_z;
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_w, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                for (int e = 0; e <= c_y; e++) {
                    p_z = l_z;
                    k_y = 1 - 0.5 * ((e == 0) | (e == c_y));
                    for (int j = 0; j <= c_z; j++) {
                        vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_w + step_w, vl), step_w, v_idx_d, vl);
                        k_z = 1 - 0.5 * ((j == 0) | (j == c_z));
                        ans_w = 0.5 * f(p_x, p_y, p_z, l_w);
                        vfloat64m2_t v_ans_w = __riscv_vfmv_v_f_f64m2(0.0, vl);
                        for (long long int k = 0; k < c_w / vl; k++) {
                            vfloat64m2_t v_ans_n = f_vector(p_x, p_y, p_z, v_point, vl);
                            v_ans_w = __riscv_vfadd_vv_f64m2(v_ans_w, v_ans_n, vl);
                            v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                        }
                        vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans_w, v_zero, vl);
                        ans_w += __riscv_vfmv_f_s_f64m1_f64(v_sum);
                        if (c_w % vl != 0) {
                            for (double point = r_w - (c_w) % vl * step_w; point < r_w; point += step_w) {
                                ans_w += f(p_x, p_y, p_z, point);
                            }
                        }
                        ans_w += 0.5 * f(p_x, p_y, p_z, r_w);
                        ans += k_x * k_y * k_z * ans_w;
                        p_z += step_z;
                    }
                    p_y += step_y;
                }
            }
        }
        return ans * step_x * step_y * step_z * step_w;
    }
#endif

    double sentr_rectangle_5(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
        double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double step_q = (r_q - l_q) / c_q;
        double ans = 0, p_y = 0, p_z = 0, p_w = 0, p_q = 0;
        double p_x = l_x + step_x / 2;;
        for (int i = 0; i < c_x; i++) {
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                p_z = l_z + step_z / 2;
                for (int k = 0; k < c_z; k++) {
                    p_w = l_w + step_w / 2;
                    for (int e = 0; e < c_w; e++) {
                        p_q = l_q + step_q / 2;
                        for (int ee = 0; ee < c_q; ee++) {
                            ans += f(p_x, p_y, p_z, p_w, p_q);
                            p_q += step_q;
                        }
                        p_w += step_w;
                    }
                    p_z += step_z;
                }
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y * step_z * step_w * step_q;
    }

    double sentr_rectangle_5_parallel(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
        double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double step_q = (r_q - l_q) / c_q;
        double ans = 0;
        double p_x = l_x + step_x / 2;
        double p_y = l_y + step_y / 2;
        double p_z = l_z + step_z / 2;
        double p_w = l_w + step_w / 2;
        double p_q = l_q + step_q / 2;
        omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : ans) private(p_x, p_y, p_z, p_w, p_q)
        for (int i = 0; i < c_x; i++) {
            p_x = l_x + step_x * (i + 0.5);
            p_y = l_y + step_y / 2;
            for (int j = 0; j < c_y; j++) {
                p_z = l_z + step_z / 2;
                for (int k = 0; k < c_z; k++) {
                    p_w = l_w + step_w / 2;
                    for (int e = 0; e < c_w; e++) {
                        p_q = l_q + step_q / 2;
                        for (int ee = 0; ee < c_q; ee++) {
                            ans += f(p_x, p_y, p_z, p_w, p_q);
                            p_q += step_q;
                        }
                        p_w += step_w;
                    }
                    p_z += step_z;
                }
                p_y += step_y;
            }
        }
        return ans * step_x * step_y * step_z * step_w * step_q;
    }

#if (defined(__riscv))
    double sentr_rectangle_5_parallel_vector(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
        double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q, int threads) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double step_q = (r_q - l_q) / c_q;
        omp_set_num_threads(threads);
        double ans = 0;
        size_t vl = __riscv_vsetvlmax_e64m2();
#pragma omp parallel reduction(+ : ans)
        {
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_y, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
#pragma omp for
            for (int i = 0; i < c_x; i++) {
                v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
                double p_x = l_x + step_x / 2.0 + i * step_x;
                for (int e = 0; e < c_w; e++) {
                    double p_w = l_w + step_w / 2.0 + e * step_w;
                    for (int ee = 0; ee < c_q; ee++) {
                        double p_q = l_q + step_q / 2.0 + ee * step_q;
                        for (int k = 0; k < c_z; k++) {
                            double p_z = l_z + step_z / 2.0 + k * step_z;
                            vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_y + step_y / 2, vl), step_y, v_idx_d, vl);
                            for (long long int j = 0; j < c_y / vl; j++) {
                                vfloat64m2_t v_ans_n = f_vector(p_x, v_point, p_z, p_w, p_q, vl);
                                v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                                v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                            }
                            if (c_y % vl != 0) {
                                for (double point = r_y - (c_y) % vl * step_y + step_y / 2; point < r_y; point += step_y) {
                                    ans += f(p_x, point, p_z, p_w, p_q);
                                }
                            }
                        }
                    }
                }
                vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
                ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
            }
        }
        return ans * step_x * step_y * step_z * step_w * step_q;
    }
#endif

    double trapezoid_5(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
        double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q) {
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double step_q = (r_q - l_q) / c_q;
        double ans = 0;
        double p_x = l_x, p_y, p_z, p_w, p_q, k_x, k_y, k_z, k_w, ans_q;
        for (int i = 0; i <= c_x; i++) {
            p_y = l_y;
            k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
            for (int j = 0; j <= c_y; j++) {
                p_z = l_z;
                k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                for (int k = 0; k <= c_z; k++) {
                    p_w = l_w;
                    k_z = 1 - 0.5 * ((k == 0) | (k == c_z));
                    for (int e = 0; e <= c_w; e++) {
                        p_q = l_q;
                        k_w = 1 - 0.5 * ((e == 0) | (e == c_w));
                        ans_q = 0.5 * f(p_x, p_y, p_z, p_w, p_q);
                        p_q += step_q;
                        for (int ee = 1; ee < c_q; ee++) {
                            ans_q += f(p_x, p_y, p_z, p_w, p_q);
                            p_q += step_q;
                        }
                        ans_q += 0.5 * f(p_x, p_y, p_z, p_w, p_q);
                        ans += k_x * k_y * k_z * k_w * ans_q;
                        p_w += step_w;
                    }
                    p_z += step_z;
                }
                p_y += step_y;
            }
            p_x += step_x;
        }
        return ans * step_x * step_y * step_z * step_w * step_q;
    }

    double trapezoid_5_parallel(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
        double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q, int threads) {
        omp_set_num_threads(threads);
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double step_q = (r_q - l_q) / c_q;
        double ans = 0;
#pragma omp parallel reduction(+ : ans)
        {
            double p_x = l_x, p_y, p_z, p_w, p_q, k_x, k_y, k_z, k_w, ans_q;
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                for (int j = 0; j <= c_y; j++) {
                    p_z = l_z;
                    k_y = 1 - 0.5 * ((j == 0) | (j == c_y));
                    for (int k = 0; k <= c_z; k++) {
                        p_w = l_w;
                        k_z = 1 - 0.5 * ((k == 0) | (k == c_z));
                        for (int e = 0; e <= c_w; e++) {
                            p_q = l_q;
                            k_w = 1 - 0.5 * ((e == 0) | (e == c_w));
                            ans_q = 0.5 * f(p_x, p_y, p_z, p_w, p_q);
                            p_q += step_q;
                            for (int ee = 1; ee < c_q; ee++) {
                                ans_q += f(p_x, p_y, p_z, p_w, p_q);
                                p_q += step_q;
                            }
                            ans_q += 0.5 * f(p_x, p_y, p_z, p_w, p_q);
                            ans += k_x * k_y * k_z * k_w * ans_q;
                            p_w += step_w;
                        }
                        p_z += step_z;
                    }
                    p_y += step_y;
                }
            }
        }
        return ans * step_x * step_y * step_z * step_w * step_q;
    }

#if (defined(__riscv))
    double trapezoid_5_parallel_vector(double l_x, double r_x, double l_y, double r_y, double l_z, double r_z, double l_w, double r_w,
        double l_q, double r_q, int c_x, int c_y, int c_z, int c_w, int c_q, int threads) {
        size_t vl = __riscv_vsetvlmax_e64m2();
        double step_x = (r_x - l_x) / c_x;
        double step_y = (r_y - l_y) / c_y;
        double step_z = (r_z - l_z) / c_z;
        double step_w = (r_w - l_w) / c_w;
        double step_q = (r_q - l_q) / c_q;
        double ans = 0;
        c_q--;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            double k_x, k_y, k_z, k_w, ans_q, p_x, p_y, p_z, p_w;
            vfloat64m2_t v_step = __riscv_vfmv_v_f_f64m2(vl * step_q, vl);
            vuint64m2_t v_idx = __riscv_vid_v_u64m2(vl);
            vfloat64m2_t v_idx_d = __riscv_vfcvt_f_xu_v_f64m2(v_idx, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
#pragma omp for
            for (int i = 0; i <= c_x; i++) {
                p_x = l_x + i * step_x;
                p_y = l_y;
                k_x = 1 - 0.5 * ((i == 0) | (i == c_x));
                for (int e = 0; e <= c_y; e++) {
                    p_z = l_z;
                    k_y = 1 - 0.5 * ((e == 0) | (e == c_y));
                    for (int ee = 0; ee <= c_z; ee++) {
                        p_w = l_w;
                        k_z = 1 - 0.5 * ((ee == 0) | (ee == c_z));
                        for (int j = 0; j <= c_w; j++) {
                            vfloat64m2_t v_point = __riscv_vfmacc_vf_f64m2(__riscv_vfmv_v_f_f64m2(l_q + step_q, vl), step_q, v_idx_d, vl);
                            k_w = 1 - 0.5 * ((j == 0) | (j == c_w));
                            ans_q = 0.5 * f(p_x, p_y, p_z, p_w, l_q);
                            vfloat64m2_t v_ans_q = __riscv_vfmv_v_f_f64m2(0.0, vl);
                            for (long long int k = 0; k < c_q / vl; k++) {
                                vfloat64m2_t v_ans_n = f_vector(p_x, p_y, p_z, p_w, v_point, vl);
                                v_ans_q = __riscv_vfadd_vv_f64m2(v_ans_q, v_ans_n, vl);
                                v_point = __riscv_vfadd_vv_f64m2(v_point, v_step, vl);
                            }
                            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans_q, v_zero, vl);
                            ans_q += __riscv_vfmv_f_s_f64m1_f64(v_sum);
                            if (c_q % vl != 0) {
                                for (double point = r_q - (c_q) % vl * step_q; point < r_q; point += step_q) {
                                    ans_q += f(p_x, p_y, p_z, p_w, point);
                                }
                            }
                            ans_q += 0.5 * f(p_x, p_y, p_z, p_w, r_q);
                            ans += k_x * k_y * k_z * k_w * ans_q;
                            p_w += step_w;
                        }
                        p_z += step_z;
                    }
                    p_y += step_y;
                }
            }
        }
        return ans * step_x * step_y * step_z * step_w * step_q;
    }
#endif

#if (defined(__riscv))
    double base_montekarlo_n(int n, std::vector<double>& l, std::vector<double>& r, long long int c) {
        double k = 1;
        double ans = 0;
        double* p = new double[n];
        for (int i = 0; i < n; i++) {
            k *= r[i] - l[i];
        }
        openrand::Philox rng(1234, 0);
        for (int i = 0; i < c; i++) {
            for (int s = 0; s < n; s++) {
                p[s] = l[s] + (rng.template rand<double>() * (r[s] - l[s]));
            }
            ans += f(p, n);
        }
        delete[] p;
        return ans * k / c;
    }

    double base_montekarlo_n_parallel(int n, std::vector<double>& l, std::vector<double>& r, long long int c, int threads) {
        double k = 1;
        double ans = 0;
        unsigned int th = threads;
        omp_set_num_threads(threads);
        for (int i = 0; i < n; i++) {
            k *= r[i] - l[i];
        }
#pragma omp parallel reduction(+ : ans)
        {
            unsigned int t = omp_get_thread_num();
            openrand::Philox rng(1234, t);
            double* p = new double[n];
            for (int i = 0; i < c / threads; i++) {
                for (int s = 0; s < n; s++) {
                    p[s] = l[s] + (rng.template rand<double>() * (r[s] - l[s]));
                }
                ans += f(p, n);
            }
            delete[] p;
        }
        return ans * k / c;
    }

    double base_montekarlo_n_parallel_vector(int n, std::vector<double>& l, std::vector<double>& r, long long int c, int threads) {
        double kf = 1;
        for (int i = 0; i < n; i++) {
            kf *= r[i] - l[i];
        }
        size_t vl = __riscv_vsetvlmax_e64m2();
        const size_t bufer_size = vl * 1000 * (n);
        double ans = 0;
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            unsigned int t = omp_get_thread_num();
            openrand::Philox rng(1234, 0, t);
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);

            double** v_p = new double* [n];
            for (int q = 0; q < n; q++) {
                v_p[q] = new double[bufer_size / n];
            }

#pragma omp for
            for (int i = 0; i < c / (bufer_size / n); i++) {
                for (int j = 0; j < bufer_size / n; j++) {
                    for (int k = 0; k < n; k++) {
                        v_p[k][j] = l[k] + (rng.template rand<double>() * (r[k] - l[k]));
                    }
                }
                for (int j = 0; j < bufer_size / n; j += vl) {
                    vfloat64m2_t v_ans_n = f_vector(v_p, n, j, vl);
                    v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                }
            }

            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
            ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
            for (int q = 0; q < n; q++) delete[] v_p[q];
            delete[] v_p;
        }
        openrand::Philox rng(1234, 0, threads + 1);
        double* p = new double[n];
        int rem = (c % (bufer_size / n));
        for (int i = 0; i < rem; i++) {
            for (int s = 0; s < n; s++) {
                p[s] = l[s] + (rng.template rand<double>() * (r[s] - l[s]));
            }
            ans += f(p, n);
        }
        delete[]p;
        return ans * kf / c;
    }

    double sobol_montekarlo_n(int n, std::vector<double>& l, std::vector<double>& r, long long int c) {
        double kf = 1;
        for (int i = 0; i < n; i++) {
            kf *= r[i] - l[i];
        }
        double ans = 0;
        const int bufer_size = 10000;
        sobol s(n, c);
        double* p = new double[bufer_size * n];
        for (int j = 0; j < c / bufer_size; j++) {
            s.sobol_points(p, bufer_size, n);
            for (int i = 0; i < bufer_size; i++) {
                for (int q = 0; q < n; q++) {
                    p[i * n + q] = l[q] + (p[i * n + q] * (r[q] - l[q]));
                }
                ans += f(p + i * n, n);
            }
        }
        delete[] p;
        if (c % bufer_size != 0) {
            double* w = new double[c % bufer_size];
            s.sobol_points(w, c % bufer_size, n);
            for (int i = 0; i < c % bufer_size; i++) {
                for (int q = 0; q < n; q++) {
                    w[i * n + q] = l[q] + (p[i * n + q] * (r[q] - l[q]));
                }
                ans += f(w + i * n, n);
            }
            delete[] w;
        }
        return ans * kf / c;
    }

    double sobol_montekarlo_n_parallel(int n, std::vector<double>& l, std::vector<double>& r, long long int c, int threads) {
        double k = 1;
        for (int i = 0; i < n; i++) {
            k *= r[i] - l[i];
        }
        double ans = 0;
        const int bufer_size = 1000 * n;
        unsigned int th = threads;
        omp_set_num_threads(threads);
        int rr = c % threads;
#pragma omp parallel reduction(+ : ans)
        {
            sobol s(n, c);
            unsigned int t = omp_get_thread_num();
            s.skip_ahead(c / threads * (t));
            int sz = c / threads;
            if (t == threads - 1) {
                sz += rr;
            }
            double* p = new double[bufer_size];
            for (int j = 0; j < sz / (bufer_size / n); j++) {
                s.sobol_points(p, bufer_size / n, n);
                for (int i = 0; i < bufer_size / n; i++) {
                    for (int q = 0; q < n; q++) {
                        p[i * n + q] = l[q] + (p[i * n + q] * (r[q] - l[q]));
                    }
                    ans += f(p + i * n, n);
                }
            }
            delete[] p;
            if (sz % (bufer_size / n) != 0) {
                double* w = new double[(sz % (bufer_size / n)) * n];
                s.sobol_points(w, sz % (bufer_size / n), n);
                for (int i = 0; i < sz % (bufer_size / n); i++) {
                    for (int q = 0; q < n; q++) {
                        w[i * n + q] = l[q] + (w[i * n + q] * (r[q] - l[q]));
                    }
                    ans += f(w + i * n, n);
                }
                delete[] w;
            }
        }
        return ans * k / c;
    }

    double sobol_montekarlo_n_parallel_vector(int n, std::vector<double>& l, std::vector<double>& r, long long int c, int threads) {
        double k = 1;
        for (int i = 0; i < n; i++) {
            k *= r[i] - l[i];
        }
        size_t vl = __riscv_vsetvlmax_e64m2();
        double ans = 0;
        const int bufer_size = 1000 * vl * n;
        omp_set_num_threads(threads);

        int rr = c % threads;
#pragma omp parallel reduction (+: ans)
        {
            sobol s(n, c);
            vfloat64m2_t v_ans = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
            unsigned int t = omp_get_thread_num();
            s.skip_ahead(c / threads * (t));
            int sz = c / threads;

            if (t == threads - 1) {
                sz += rr;
            }
            double** v_p = new double* [n];
            for (int q = 0; q < n; q++) {
                v_p[q] = new double[bufer_size / n];
            }
            double* p = new double[bufer_size];
            for (int j = 0; j < sz * n / bufer_size; j++) {
                s.sobol_points(p, bufer_size / n, n);
                for (int i = 0; i < bufer_size / n; i++) {
                    for (int q = 0; q < n; q++) {
                        v_p[q][i] = l[q] + (p[i * n + q] * (r[q] - l[q]));
                    }
                }
                for (int i = 0; i < bufer_size / n; i += vl) {
                    vfloat64m2_t v_ans_n = f_vector(v_p, n, i, vl);
                    v_ans = __riscv_vfadd_vv_f64m2(v_ans, v_ans_n, vl);
                }
            }
            delete[] p;
            for (int q = 0; q < n; q++) delete[] v_p[q];
            delete[] v_p;

            vfloat64m1_t v_sum = __riscv_vfredusum_vs_f64m2_f64m1(v_ans, v_zero, vl);
            ans += __riscv_vfmv_f_s_f64m1_f64(v_sum);
            int rem = (sz % (bufer_size / n));
            if (rem != 0) {
                double* ww = new double[rem * n];
                s.sobol_points(ww, rem, n);
                double* w = new double[n];
                for (int i = 0; i < rem; i++) {
                    for (int q = 0; q < n; q++) {
                        w[q] = l[q] + (ww[i * n + q] * (r[q] - l[q]));
                    }
                    ans += f(w, n);
                }
                delete[] w;
                delete[]ww;
            }
        }
        return ans * k / c;
    }

#else

    double base_montekarlo_n(int n, std::vector<double>& l, std::vector<double>& r, long long int c) {
        double ans = 0;
        const int bufer_sz = 1000;
        double* p = new double[n * bufer_sz];
        double k = 1;
        for (int i = 0; i < n; i++) {
            k *= (r[i] - l[i]);
        }
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, 777);
        for (int i = 0; i < c / bufer_sz; i++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, bufer_sz * n, p, 0, 1);
            for (int j = 0; j < bufer_sz; j++) {
                for (int s = 0; s < n; s++) {
                    p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                }
            }
            for (int j = 0; j < bufer_sz; j++) {
                ans += f(p + n * j, n);
            }
        }
        int rem = c % bufer_sz;
        if (rem != 0) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, rem * n, p, 0, 1);
            for (int j = 0; j < rem; j++) {
                for (int s = 0; s < n; s++) {
                    p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                }
            }
            for (int j = 0; j < rem; j++) {
                ans += f(p + n * j, n);
            }
        }
        vslDeleteStream(&stream);
        delete[] p;
        return k * ans / c;
    }

    double base_montekarlo_n_parallel(int n, std::vector<double>& l, std::vector<double>& r, long long int c, int threads) {
        const int bufer_sz = 1000;
        double ans = 0;
        double k = 1;
        for (int i = 0; i < n; i++) {
            k *= (r[i] - l[i]);
        }
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SFMT19937, 777);
        const int th = threads;
        VSLStreamStatePtr* streams = new VSLStreamStatePtr[th];
        for (int i = 0; i < th; i++) {
            vslCopyStream(&streams[i], stream);
            vslSkipAheadStream(streams[i], i * n * c / th);
        }
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            int t = omp_get_thread_num();
            double* p = new double[bufer_sz * n];
            for (int i = 0; i < c / th / bufer_sz; i++) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], bufer_sz * n, p, 0, 1);
                for (int j = 0; j < bufer_sz; j++) {
                    for (int s = 0; s < n; s++) {
                        p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                    }
                }
                for (int j = 0; j < bufer_sz; j++) {
                    ans += f(p + n * j, n);
                }
            }
            const int rem = (c / th) % bufer_sz;
            if (rem != 0) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], rem * n, p, 0, 1);
                for (int j = 0; j < rem; j++) {
                    for (int s = 0; s < n; s++) {
                        p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                    }
                }
                for (int j = 0; j < rem; j++) {
                    ans += f(p + n * j, n);
                }
            }
            delete[] p;
        }
        for (int i = 0; i < th; i++) {
            vslDeleteStream(&streams[i]);
        }
        vslDeleteStream(&stream);
        delete[] streams;
        return ans * k / c;
    }

    double sobol_montekarlo_n(int n, std::vector<double>& l, std::vector<double>& r, long long int c) {
        double ans = 0;
        const int bufer_sz = 1000;
        double* p = new double[n * bufer_sz];
        double k = 1;
        for (int i = 0; i < n; i++) {
            k *= (r[i] - l[i]);
        }
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SOBOL, n);
        for (int i = 0; i < c / bufer_sz; i++) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, bufer_sz * n, p, 0, 1);
            for (int j = 0; j < bufer_sz; j++) {
                for (int s = 0; s < n; s++) {
                    p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                }
            }
            for (int j = 0; j < bufer_sz; j++) {
                ans += f(p + n * j, n);
            }
        }
        int rem = c % bufer_sz;
        if (rem != 0) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, rem * n, p, 0, 1);
            for (int j = 0; j < rem; j++) {
                for (int s = 0; s < n; s++) {
                    p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                }
            }
            for (int j = 0; j < rem; j++) {
                ans += f(p + n * j, n);
            }
        }
        vslDeleteStream(&stream);
        delete[] p;
        return k * ans / c;
    }

    double sobol_montekarlo_n_parallel(int n, std::vector<double>& l, std::vector<double>& r, long long int c, int threads) {
        const int bufer_sz = 1000;
        double ans = 0;
        double k = 1;
        for (int i = 0; i < n; i++) {
            k *= (r[i] - l[i]);
        }
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_SOBOL, n);
        const int th = threads;
        VSLStreamStatePtr* streams = new VSLStreamStatePtr[th];
        for (int i = 0; i < th; i++) {
            vslCopyStream(&streams[i], stream);
            vslSkipAheadStream(streams[i], i * (c / th));
        }
        omp_set_num_threads(threads);
#pragma omp parallel reduction(+ : ans)
        {
            int t = omp_get_thread_num();
            double* p = new double[bufer_sz * n];
            for (int i = 0; i < c / th / bufer_sz; i++) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], bufer_sz * n, p, 0, 1);
                for (int j = 0; j < bufer_sz; j++) {
                    for (int s = 0; s < n; s++) {
                        p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                    }
                }
                for (int j = 0; j < bufer_sz; j++) {
                    ans += f(p + n * j, n);
                }
            }
            const int rem = (c / th) % bufer_sz;
            if (rem != 0) {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[t], rem * n, p, 0, 1);
                for (int j = 0; j < rem; j++) {
                    for (int s = 0; s < n; s++) {
                        p[j * n + s] = p[j * n + s] * (r[s] - l[s]) + l[s];
                    }
                }
                for (int j = 0; j < rem; j++) {
                    ans += f(p + n * j, n);
                }
            }
            delete[] p;
        }
        for (int i = 0; i < th; i++) {
            vslDeleteStream(&streams[i]);
        }
        delete[] streams;
        vslDeleteStream(&stream);
        return ans * k / c;
    }
#endif

};



double ffff(double* v, long long int size) {
    double ans = 1;
    for (int i = 0; i < size; i++) {
        ans *= exp(v[i]);
        ans *= exp(v[i] + 1);
        ans *= exp(v[i] + 2);
        ans *= exp(v[i] + 3);
        ans *= exp(v[i] + 4);
        ans *= exp(v[i] + 5);
        ans *= exp(v[i] + 6);
        ans *= exp(v[i] + 7);
        ans *= exp(v[i] + 8);
        ans *= exp(v[i] + 9);
    }
    return ans;
}

vfloat64m2_t fff(double** v, int n, int j, size_t vl) {
    vfloat64m2_t ans = __riscv_vfmv_v_f_f64m2(1.0, vl);
    vfloat64m2_t v_1 = __riscv_vfmv_v_f_f64m2(1.0, vl);
    for (int i = 0; i < n; i++) {
        vfloat64m2_t vec = __riscv_vle64_v_f64m2(v[i] + j, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
        vec = __riscv_vfadd_vv_f64m2(vec, v_1, vl);
        ans = __riscv_vfmul_vv_f64m2(ans, __riscv_vexp_f64m2(vec, vl), vl);
    }
    return ans;
}

int main(int argc, char* argv[])
{
    std::string key = argv[1];
    int cp_min = std::stoi(argv[2]);
    int cp_max = std::stoi(argv[3]);
    int step = std::stoi(argv[4]);
    int n = std::stoi(argv[5]);
    int flag = std::stoi(argv[6]);
    int threads = std::stoi(argv[7]);
    double ans = 0;
    double global_ans = 0;
    if (flag) {
        flag = 0;
    }
    else {
        auto begin = std::chrono::steady_clock::now();
        if (cp_max != cp_min) {
            std::cout << cp_max << " " << cp_min << " " << flag << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        }
        else {
            int c_p = cp_min;
            if (n == 1) {
                auto m = montecarlo([](double x) { return std::exp(x); }, [](vfloat64m2_t x, size_t vl) {return __riscv_vexp_f64m2(x, vl); });
                if (key == "rr") {
                    ans = m.right_rectangle(0, 1, c_p);
                }
                if (key == "lr") {
                    ans = m.left_rectangle(0, 1, c_p);
                };
                if (key == "sr") {
                    ans = m.sentr_rectangle(0, 1, c_p);
                }
                if (key == "t") {
                    ans = m.trapezoid(0, 1, c_p);
                }
                if (key == "m") {
                    ans = m.base_montekarlo(0, 1, c_p);
                }
                if (key == "sm") {
                    ans = m.sobol_montekarlo(0, 1, c_p);

                }
                if (key == "rrp") {
                    ans = m.right_rectangle_parallel(0, 1, c_p, threads);
                }
                if (key == "lrp") {
                    ans = m.left_rectangle_parallel(0, 1, c_p, threads);
                };
                if (key == "srp") {
                    ans = m.sentr_rectangle_parallel(0, 1, c_p, threads);
                }
                if (key == "tp") {
                    ans = m.trapezoid_parallel(0, 1, c_p, threads);
                }
                if (key == "mp") {
                    ans = m.base_montekarlo_parallel(0, 1, c_p, threads);
                }
                if (key == "smp") {
                    ans = m.sobol_montekarlo_parallel(0, 1, c_p, threads);
                }

                if (key == "srpv") {
                    ans = m.sentr_rectangle_parallel_vector(0, 1, c_p, threads);
                }
                if (key == "tpv") {
                    ans = m.trapezoid_parallel_vector(0, 1, c_p, threads);
                }
                if (key == "mpv") {
                    ans = m.base_montekarlo_parallel_vector(0, 1, c_p, threads);
                }
                if (key == "smpv") {
                    ans = m.sobol_montekarlo_parallel_vector(0, 1, c_p, threads);
                }
            }

            if (n == 2) {
                auto func1{ [](double x, double y) {return exp(x) * exp(y); } };
                auto func2{ [](double x, vfloat64m2_t y, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(y, vl), exp(x), vl);  } };
                double right_ans = (exp(1) - 1) * (exp(1) - 1);
                auto m = montecarlo(func1, func2);
                if (key == "sr") {
                    int c = (int)pow(c_p, 1.0 / 2);
                    ans = m.sentr_rectangle_2(0, 1, 0, 1, c, c);
                }
                if (key == "t") {
                    int c = (int)pow(c_p, 1.0 / 2);
                    ans = m.trapezoid_2(0, 1, 0, 1, c, c);
                }
                if (key == "srp") {
                    int c = (int)pow(c_p, 1.0 / 2);
                    ans = m.sentr_rectangle_2_parallel(0, 1, 0, 1, c, c, threads);
                }
                if (key == "tp") {
                    int c = (int)pow(c_p, 1.0 / 2);
                    ans = m.trapezoid_2_parallel(0, 1, 0, 1, c, c, threads);
                }


                if (key == "srpv") {
                    int c = (int)pow(c_p, 1.0 / 2);
                    ans = m.sentr_rectangle_2_parallel_vector(0, 1, 0, 1, c, c, threads);
                }
                if (key == "tpv") {
                    int c = (int)pow(c_p, 1.0 / 2);
                    ans = m.trapezoid_2_parallel_vector(0, 1, 0, 1, c, c, threads);
                }
            }

            if (n == 3) {
                auto func1{ [](double x, double y, double z) {return exp(x) * exp(y) * exp(z); } };
                auto func2{ [](double x, vfloat64m2_t y, double z, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(y, vl), exp(x) * exp(z), vl);  } };
                auto func3{ [](double x, double y, vfloat64m2_t z, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(z, vl), exp(x) * exp(y), vl);  } };
                double right_ans = (exp(1) - 1) * (exp(1) - 1) * (exp(1) - 1);
                auto m = montecarlo(func1, func2);
                if (key == "sr") {
                    int c = (int)pow(c_p, 1.0 / 3);
                    ans = m.sentr_rectangle_3(0, 1, 0, 1, 0, 1, c, c, c);
                }
                if (key == "t") {
                    int c = (int)pow(c_p, 1.0 / 3);
                    ans = m.trapezoid_3(0, 1, 0, 1, 0, 1, c, c, c);
                }
                if (key == "srp") {
                    int c = (int)pow(c_p, 1.0 / 3);
                    ans = m.sentr_rectangle_3_parallel(0, 1, 0, 1, 0, 1, c, c, c, threads);
                }
                if (key == "tp") {
                    int c = (int)pow(c_p, 1.0 / 3);
                    ans = m.trapezoid_3_parallel(0, 1, 0, 1, 0, 1, c, c, c, threads);
                }

                if (key == "srpv") {
                    int c = (int)pow(c_p, 1.0 / 3);
                    ans = m.sentr_rectangle_3_parallel_vector(0, 1, 0, 1, 0, 1, c, c, c, threads);
                }

                auto m2 = montecarlo(func1, func3);
                if (key == "tpv") {
                    int c = (int)pow(c_p, 1.0 / 3);
                    ans = m2.trapezoid_3_parallel_vector(0, 1, 0, 1, 0, 1, c, c, c, threads);
                }
            }

            if (n == 4) {
                auto func1{ [](double x, double y, double z, double w) {return exp(x) * exp(y) * exp(z) * exp(w); } };
                auto func2{ [](double x, vfloat64m2_t y, double z, double w, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(y, vl), exp(x) * exp(z) * exp(w), vl);  } };
                auto func3{ [](double x, double y, double z, vfloat64m2_t w, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(w, vl), exp(x) * exp(y) * exp(z), vl);  } };
                double right_ans = (exp(1) - 1) * (exp(1) - 1) * (exp(1) - 1) * (exp(1) - 1);
                auto m = montecarlo(func1, func2);
                if (key == "sr") {
                    int c = (int)pow(c_p, 1.0 / 4);
                    ans = m.sentr_rectangle_4(0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c);
                }
                if (key == "t") {
                    int c = (int)pow(c_p, 1.0 / 4);
                    ans = m.trapezoid_4(0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c);
                }
                if (key == "srp") {
                    int c = (int)pow(c_p, 1.0 / 4);
                    ans = m.sentr_rectangle_4_parallel(0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, threads);
                }
                if (key == "tp") {
                    int c = (int)pow(c_p, 1.0 / 4);
                    ans = m.trapezoid_4_parallel(0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, threads);
                }

                if (key == "srpv") {
                    int c = (int)pow(c_p, 1.0 / 4);
                    ans = m.sentr_rectangle_4_parallel_vector(0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, threads);
                }

                auto m2 = montecarlo(func1, func3);
                if (key == "tpv") {
                    int c = (int)pow(c_p, 1.0 / 4);
                    ans = m2.trapezoid_4_parallel_vector(0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, threads);
                }
            }

            if (n == 5) {
                auto func1{ [](double x, double y, double z, double w, double q) {return exp(x) * exp(y) * exp(z) * exp(w) * exp(q); } };
                auto func2{ [](double x, vfloat64m2_t y, double z, double w, double q, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(y, vl), exp(x) * exp(z) * exp(w) * exp(q), vl);  } };
                auto func3{ [](double x, double y, double z, double w, vfloat64m2_t q, size_t vl) { return __riscv_vfmul_vf_f64m2(__riscv_vexp_f64m2(q, vl), exp(x) * exp(y) * exp(z) * exp(w),  vl);  } };
                double right_ans = (exp(1) - 1) * (exp(1) - 1) * (exp(1) - 1) * (exp(1) - 1) * (exp(1) - 1);
                auto m = montecarlo(func1, func2);
                if (key == "sr") {
                    int c = (int)pow(c_p, 1.0 / 5);
                    ans = m.sentr_rectangle_5(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, c);
                }
                if (key == "t") {
                    int c = (int)pow(c_p, 1.0 / 5);
                    ans = m.trapezoid_5(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, c);
                }
                if (key == "srp") {
                    int c = (int)pow(c_p, 1.0 / 5);
                    ans = m.sentr_rectangle_5_parallel(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, c, threads);
                }
                if (key == "tp") {
                    int c = (int)pow(c_p, 1.0 / 5);
                    ans = m.trapezoid_5_parallel(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, c, threads);
                }

                if (key == "srpv") {
                    int c = (int)pow(c_p, 1.0 / 5);
                    ans = m.sentr_rectangle_5_parallel_vector(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, c, threads);
                }

                auto m2 = montecarlo(func1, func3);
                if (key == "tpv") {
                    int c = (int)pow(c_p, 1.0 / 5);
                    ans = m2.trapezoid_5_parallel_vector(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, c, c, c, c, c, threads);
                }
            }

            if (((key == "m") || (key == "mp") || (key == "sm") || (key == "smp")) || (key == "smpv") || (key == "mpv") && (n != 1)) {
                auto lambda_multidim = [](double* v, long long int size) { return ffff(v, size); };

                auto lambda_multidim_vector = [](double** v, int n, int j, size_t vl) { return fff(v, n, j, vl); };
                auto m = montecarlo(lambda_multidim, lambda_multidim_vector);
                double right_ans = 1;
                auto l = std::vector<double>(n);
                auto r = std::vector<double>(n);
                auto c = std::vector<int>(n);
                for (int i = 0; i < n; i++) {
                    l[i] = 0;
                    r[i] = 1;
                    right_ans *= (exp(1) - 1);
                }
                if (key == "m") {
                    ans = m.base_montekarlo_n(n, l, r, c_p);
                }
                if (key == "mp") {
                    ans = m.base_montekarlo_n_parallel(n, l, r, c_p, threads);
                }
                if (key == "sm") {
                    ans = m.sobol_montekarlo_n(n, l, r, c_p);
                }
                if (key == "smp") {
                    ans = m.sobol_montekarlo_n_parallel(n, l, r, c_p, threads);
                }
                if (key == "mpv") {
                    ans = m.base_montekarlo_n_parallel_vector(n, l, r, c_p, threads);
                }
                if (key == "smpv") {
                    ans = m.sobol_montekarlo_n_parallel_vector(n, l, r, c_p, threads);
                }
            }
        }

        if (ans == 0) {
            global_ans++;
        }

        auto end = std::chrono::steady_clock::now();
        auto el = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << key << " " << cp_min << " " << el.count() << " " << ans << "";
    }
    return 0;
}
