/*
    meow_fft. My Easy Oresome Wonderful Fast Fourier Transform.
    Copyright (C) 2017 Richard Maxwell <jodi.the.tigger@gmail.com>
    This file is part of meow_fft
    meow_fft is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>
*/

#define MEOW_FFT_IMPLEMENTATION
#include "meow_fft.h"

#include <kiss_fftr.h>
#include <pffft.h>

#ifdef TEST_WITH_FFTW3
// FFTW testing
#include <fftw3.h>
#endif

#include <vector>
#include <array>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------------------

// Some things I like about rust.
#define let const auto
#define var auto
#define fun auto

using namespace std;

fun test
(
      const float*            in
    , const float*            out
    , const unsigned          data_count
    , const Meow_FFT_Complex* fft
    , const Meow_FFT_Complex* fft_reference
    , const unsigned          fft_count
    , float                   epislon
)
-> bool
{

    if (!in || !out || !fft)
    {
        printf("Invalid inputs\n");
        return false;
    }

    if (fft_count < (1 + (data_count / 2)))
    {
        printf("fft too small\n");
        return false;
    }

    bool  win           = true;
    float worst_epislon = 0.0f;
    bool  odd_size      = (data_count & 1);
    bool  full_fft      = (fft_count == data_count);

    // -------------------------------------------------------------------------

    // fft[0]   == sum of inputs
    // fft[N/2] == input dot [1,-1,1,-1,1,-1...]
    float sum = 0;
    float dot = 0;

    for (unsigned i = 0; i < data_count; i++)
    {
        let a = in[i];
        let b = out[i] / data_count;

        worst_epislon = max(abs(a - b), worst_epislon);

        if ((win) && (worst_epislon > epislon))
        {
            printf
            (
                  "= %d ===: In/Out         : %10.7f != %10.7f (%d)\n"
                , data_count
                , a
                , b
                , i
            );

            win = false;
        }

        sum += a;
        dot += a * (1.0f - ((i & 1) * 2));
    }

    if (!win)
    {
        printf
        (
              "= %d ===: In/Out epislon : %10.7f\n"
            , data_count
            , worst_epislon
        );
    }

    // -------------------------------------------------------------------------

    let near = [epislon](float lhs, float rhs) -> bool
    {
        return abs(lhs - rhs) < epislon;
    };

    if (!near(sum, fft[0].r))
    {
        printf
        (
              "= %d ===: fft[0]/sum     : %10.7f != %10.7f (%10.7f)\n"
            , data_count
            , fft[0].r
            , sum
            , abs(fft[0].r - sum)
        );

        win = false;
    }

    if (!odd_size)
    {
        let fft_dot = (full_fft) ? fft[data_count / 2].r : fft[0].j;

        // even only sized ffts have this real only value.
        if (!near(dot, fft_dot))
        {
            printf
            (
                  "= %d ===: fft[N/2]/dot   : %10.7f != %10.7f (%10.7f)\n"
                , data_count
                , fft_dot
                , dot
                , abs(fft_dot - dot)
            );

            win = false;
        }
    }

    let meow_fft_complex_near = [near]
    (
          const Meow_FFT_Complex& a
        , const Meow_FFT_Complex& b
        , unsigned size_in
        , unsigned i
        , const char* tag = ""
    )
    -> bool
    {
        var result = true;

        if (!near(a.r, b.r))
        {
            printf
            (
                  "= %d ===: %s : r: %10.7f != %10.7f (%d)\n"
                , size_in
                , tag
                , a.r
                , b.r
                , i
            );

            result = false;
        }
        if (!near(a.j, -b.j))
        {
            printf
            (
                  "= %d ===: %s : j: %10.7f != %10.7f (%d)\n"
                , size_in
                , tag
                , a.j
                , -b.j
                , i
            );

            result = false;
        }

        return result;
    };

    if (full_fft && (!odd_size))
    {
        worst_epislon = 0.0f;

        // real inputs
        // fft[n] == conjugate(fft[N-n])
        for (unsigned i = 1; i < (data_count / 2) - 1; i++)
        {
            let a = fft[i];
            let b = fft[data_count - i];

            worst_epislon = max(abs(a.r - b.r), worst_epislon);
            worst_epislon = max(abs(a.j + b.j), worst_epislon);

            if (!meow_fft_complex_near(a, b, data_count, i, "fft[n] / [N-n]"))
            {
                win = false;
                break;
            }
        }

        if (!win)
        {
            printf
            (
                  "= %d ===: conj epislon   : %10.7f\n"
                , data_count
                , worst_epislon
            );
        }
    }

    if (fft_reference)
    {
        worst_epislon = 0.0f;

        let size_ref = fft_count;
        for (unsigned i = 0; i < size_ref; i++)
        {
            let a = fft[i];
            let b = fft_reference[i];

            worst_epislon = max(abs(a.r - b.r), worst_epislon);
            worst_epislon = max(abs(a.j - b.j), worst_epislon);

            if (!meow_fft_complex_near(a, b, data_count, i, "fft[n]!=ref[n]"))
            {
                win = false;
                break;
            }
        }

        if (!win)
        {
            printf
            (
                  "= %d ===: ref epislon    : %10.7f\n"
                , data_count
                , worst_epislon
            );
        }
    }

    if (!win)
    {
        // test thingie.
        let max = min(data_count, 10u);

        printf
        (
              "= %d ===: epislon        : %10.7f\n"
            , data_count
            , epislon
        );

        printf("in        : ");
        for (unsigned j = 0; j < max; ++j)
        {
            printf("%9.6f,", in[j]);
        }
        printf("\n");

        printf("out       : ");
        for (unsigned j = 0; j < max; ++j)
        {
            printf("%9.6f,", out[j] / data_count);
        }
        printf("\n");

        printf("fft_real  : ");
        for (unsigned j = 0; j < max; ++j)
        {
            printf("%9.6f,", fft[j].r);
        }
        printf("\n");

        if (fft_reference)
        {
            let ref_count = min(max, fft_count);
            printf("fft_real_r: ");
            for (unsigned j = 0; j < ref_count; ++j)
            {
                printf("%9.6f,", fft_reference[j].r);
            }
            printf("\n");
        }

        printf("fft_imag  : ");
        for (unsigned j = 0; j < max; ++j)
        {
            printf("%9.6f,", fft[j].j);
        }
        printf("\n");

        if (fft_reference)
        {
            let ref_count = min(max, fft_count);
            printf("fft_imag_r: ");
            for (unsigned j = 0; j < ref_count; ++j)
            {
                printf("%9.6f,", fft_reference[j].j);
            }
            printf("\n");
        }

        printf("fft_mag   : ");
        for (unsigned j = 0; j < max; ++j)
        {
            let mag = sqrt(fft[j].r * fft[j].r + fft[j].j * fft[j].j);
            printf("%9.6f,", mag);
        }
        printf("\n");
    }

    return win;
}

int main(int argc, char**)
{
    let profiling = (argc > 1) ? true : false;

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------
    if (!profiling)
    {
        static const float F_EPISLON = (2.0f / 65536);
        array<float, 2048> test_data;
        var temp = vector<Meow_FFT_Complex>(test_data.size());

        // Data from:
        // http://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
        let size = test_data.size();
        for (unsigned i = 0; i < size; ++i)
        {
            for (unsigned i = 0; i < size; ++i)
            {
                let two_pi_i = 2.0f * M_PI * i;

                test_data[i] =
                      1.0f * sin(50.0f * two_pi_i * (1.0f / 799.0f))
                    + 0.5f * sin(80.0f * two_pi_i * (1.0f / 799.0f));
            }
        }

        // ---------------------------------------------------------------------

        // First test, forward fft. Test up to 3 stage.
        printf("Testing real -> fft against reference...\n");
        fflush(stdout);

        let near = [](float lhs, float rhs) -> bool
        {
            return abs(lhs - rhs) < 0.0001f;
        };

        for (unsigned i = 4; i < 65; i += 2)
        {
            // Meow ------------------------------------------------------------
            var fft_meow = vector<Meow_FFT_Complex>(i);
            var fft_meow_a = fft_meow.data();

            let plan_meow_bytes = meow_fft_generate_workset_real(i, nullptr);

            Meow_FFT_Workset_Real* plan_meow =
                (Meow_FFT_Workset_Real*) malloc(plan_meow_bytes);

            meow_fft_generate_workset_real(i, plan_meow);
            meow_fft_real(plan_meow, test_data.data(), fft_meow_a);

            free(plan_meow);

            // Kiss ------------------------------------------------------------
            var fft_kiss = vector<Meow_FFT_Complex>(i);
            var fft_kiss_a = reinterpret_cast<kiss_fft_cpx*>(fft_kiss.data());

            let plan_kiss = kiss_fftr_alloc(i, 0, NULL, NULL);
            kiss_fftr(plan_kiss, test_data.data(), fft_kiss_a);

            kiss_fftr_free(plan_kiss);

            // Pfft ------------------------------------------------------------
            float* fft_pfft_a = nullptr;

            if (!(i % 32))
            {
                let plan_pfft  = pffft_new_setup(i, PFFFT_REAL);

                fft_pfft_a = (float*) pffft_aligned_malloc
                (
                    sizeof(float) * 2 * i
                );

                pffft_transform_ordered
                (
                      plan_pfft
                    , test_data.data()
                    , fft_pfft_a
                    , NULL
                    , PFFFT_FORWARD
                );
            }

            // FFTW ------------------------------------------------------------
            float* fft_fftw_a = nullptr;

#ifdef TEST_WITH_FFTW3
            // for some reason fftw stomps over the input array when you make
            // the plan.
            var* copy  = (float*) fftwf_malloc(sizeof(float) * i);
            fft_fftw_a = (float*) fftwf_malloc(sizeof(float) * i * 2);

            let plan_fftw = fftwf_plan_dft_r2c_1d
            (
                  i
                , copy
                , (fftwf_complex*) fft_fftw_a
                , FFTW_MEASURE
            );

            memcpy(copy, test_data.data(), i * sizeof(float));

            fftwf_execute(plan_fftw);
            fftwf_free(copy);

            fftwf_destroy_plan(plan_fftw);
#else
            auto fftwf_free = [](void*) {};
#endif


            // and now compare, but not the first and last item as
            // they are ignored and packed.
            unsigned failed = 0;
            unsigned j;
            for (j = 1; j < (i / 2); j++)
            {
                if (!near(fft_meow_a[j].r, fft_kiss_a[j].r))
                {
                    failed = 1;
                }

                if (!near(fft_meow_a[j].j, fft_kiss_a[j].i))
                {
                    failed = 2;
                }

                if (fft_fftw_a)
                {
                    if (!near(fft_meow_a[j].r, fft_fftw_a[j * 2]))
                    {
                        failed = 1;
                    }
                    if (!near(fft_meow_a[j].j, fft_fftw_a[j * 2 + 1]))
                    {
                        failed = 2;
                    }
                }

                if (failed) { break; }
            }

            if (failed)
            {
                auto dump = [i](const char* name, float* source, unsigned im)
                {
                    if (source)
                    {
                        printf("\n%s, ", name);
                        for (unsigned j = 0; j < (i / 2); j++)
                        {
                            printf("%10.7f, ", source[im + (j * 2)]);
                        }
                    }
                };

                printf
                (
                      "Failed at N %d, index %d (%s)\n"
                    , i
                    , j
                    , (failed == 1) ? "real" : "imag"
                );

                printf("real:");
                dump("meow", (float*) fft_meow_a, 0);
                dump("kiss", (float*) fft_kiss_a, 0);
                dump("pfft", (float*) fft_pfft_a, 0);
                dump("fftw", (float*) fft_fftw_a, 0);

                // now j
                printf("\n\nimag:");
                dump("meow", (float*) fft_meow_a, 1);
                dump("kiss", (float*) fft_kiss_a, 1);
                dump("pfft", (float*) fft_pfft_a, 1);
                dump("fftw", (float*) fft_fftw_a, 1);

                printf("\n");
            }

            // Free stuff
            if (fft_pfft_a) { pffft_aligned_free(fft_pfft_a); }
            if (fft_fftw_a) { fftwf_free        (fft_fftw_a); }

            if (failed) { break; }
        }

        // ---------------------------------------------------------------------

        printf("Testing real only <-> fft (skipping slow ffts)...\n");
        fflush(stdout);

        for (unsigned i = 4; i < test_data.size(); i += 2)
        {
            let workset_bytes = meow_fft_generate_workset_real(i, nullptr);

            Meow_FFT_Workset_Real* fft =
                (Meow_FFT_Workset_Real*) malloc(workset_bytes);

            meow_fft_generate_workset_real(i, fft);

            if (meow_fft_is_slow_real(fft))
            {
                free(fft);
                continue;
            }

            var f = vector<Meow_FFT_Complex>(i);

            vector<float> result(i);

            meow_fft_real(fft, test_data.data(), f.data());
            meow_fft_real_i(fft, f.data(), temp.data(), result.data());

            let win = test
            (
                  test_data.data()
                , result.data()
                , i
                , f.data()
                , nullptr
                , (i / 2) + 1
                , F_EPISLON
            );

            free(fft);

            if (!win) break;
        }
    }

    // -------------------------------------------------------------------------
    // Profiling
    // -------------------------------------------------------------------------
    // C/C++ makes it too hard to remove this duplicate code. So just copypasta.
    {
        let seconds = 5;
        let runs    = 5;

        let test_N =
        {
              64,  256, 512, 1024, 2048, 4096, 8192, 16384, 32768
            , 100, 200, 500, 1000, 1200, 5760, 10000
        };

        array<float, 44100 * seconds> speed_data;
        let size = speed_data.size();        

        for (unsigned i = 0; i < size; ++i)
        {
            let two_pi_i = 2.0f * M_PI * i;

            speed_data[i] =
                  1.0f * sin(50.0f * two_pi_i * (1.0f / 799.0f))
                + 0.5f * sin(80.0f * two_pi_i * (1.0f / 799.0f));
        }

        printf
        (
              "Profiling N fft, over %d seconds of 44.1khz data. "
              "Median value of %d runs\n"
            , seconds
            , runs
        );

        let quick_test_64 = [&speed_data]
        (
              const float* test
            , unsigned     N
        )
        {
            float worst = 0.0f;

            for (unsigned i = 0 ; i < 64; i++)
            {
                let norm = (test[i] / N);
                let diff = std::abs(speed_data[i] - norm);

                worst = (worst > diff) ? worst : diff;
            }

            printf("%5.3f, 16 bit error, ", worst * 65536.0f);
        };


        {
            for (let N : test_N)
            {
                let workset_bytes = meow_fft_generate_workset_real(N, nullptr);

                Meow_FFT_Workset_Real* fft_real =
                    (Meow_FFT_Workset_Real*) malloc(workset_bytes);

                meow_fft_generate_workset_real(N, fft_real);

                printf(" meow_fft, %7.2f, kb, ", workset_bytes / 1024.0f);

                // -------------------------------------------------------------

                var f = vector<Meow_FFT_Complex>(N);
                var t = vector<Meow_FFT_Complex>(N);

                vector<float> result(N);

                array<chrono::milliseconds, runs> values;

                {
                    meow_fft_real(fft_real, &speed_data[0], f.data());
                    meow_fft_real_i
                    (
                          fft_real
                        , f.data()
                        , t.data()
                        , result.data()
                    );

                    quick_test_64(result.data(), N);
                }

                for (unsigned sample = 0; sample < runs; sample++)
                {
                    let start = chrono::high_resolution_clock::now();

                    for (unsigned offset = 0 ; offset < (size - N); offset+=32)
                    {
                        meow_fft_real(fft_real, &speed_data[offset], f.data());
                        meow_fft_real_i
                        (
                              fft_real
                            , f.data()
                            , t.data()
                            , result.data()
                        );
                    }

                    let end = chrono::high_resolution_clock::now();
                    let delta = end - start;
                    values[sample] =
                        chrono::duration_cast<chrono::milliseconds>(delta);

                    printf(".");
                    fflush(stdout);
                }

                sort(values.begin(), values.end());

                float microseconds_per_fft_multiplier =
                    1000.0f / ((size - N) / 32);

                printf
                (
                      ", %5d, %6zd, ms, %7.2f, microseconds per fft\n"
                    , N
                    , (size_t) values[runs / 2].count()
                    , values[runs / 2].count() * microseconds_per_fft_multiplier
                );

                free(fft_real);

                fflush(stdout);
            }
        }

        {
            for (let N : test_N)
            {
                let plan_fft  = kiss_fftr_alloc(N, 0, NULL, NULL);
                let plan_ifft = kiss_fftr_alloc(N, 1, NULL, NULL);


                var* fft =
                    (kiss_fft_cpx*) malloc
                    (
                        sizeof(kiss_fft_cpx) * (1 + (N / 2))
                    );

                var* result = (float*) malloc(sizeof(float) * N);


                printf(" kiss_fft, %7.2f, kb, ", 0.0f);

                // -------------------------------------------------------------

                array<chrono::milliseconds, runs> values;

                {
                    kiss_fftr(plan_fft, &speed_data[0], fft);
                    kiss_fftri(plan_ifft, fft, result);

                    quick_test_64(result, N);
                }

                for (unsigned sample = 0; sample < runs; sample++)
                {
                    let start = chrono::high_resolution_clock::now();

                    for (unsigned offset = 0 ; offset < (size - N); offset+=32)
                    {
                        kiss_fftr(plan_fft, &speed_data[offset], fft);
                        kiss_fftri(plan_ifft, fft, result);
                    }

                    let end = chrono::high_resolution_clock::now();
                    let delta = end - start;
                    values[sample] =
                        chrono::duration_cast<chrono::milliseconds>(delta);

                    printf(".");
                    fflush(stdout);
                }

                sort(values.begin(), values.end());

                float microseconds_per_fft_multiplier =
                    1000.0f / ((size - N) / 32);

                printf
                (
                      ", %5d, %6zd, ms, %7.2f, microseconds per fft\n"
                    , N
                    , (size_t) values[runs / 2].count()
                    , values[runs / 2].count() * microseconds_per_fft_multiplier
                );

                kiss_fftr_free(plan_fft);
                kiss_fftr_free(plan_ifft);
                free(fft);
                free(result);

                fflush(stdout);
            }
        }


        {
            for (let N : test_N)
            {
                // pffft doesn't do everything.
                if (N == 100)
                {
                    continue;
                }
                if (N == 200)
                {
                    continue;
                }
                if (N == 500)
                {
                    continue;
                }
                if (N == 1000)
                {
                    continue;
                }
                if (N == 1200)
                {
                    continue;
                }
                if (N == 10000)
                {
                    continue;
                }

                let plan_fft  = pffft_new_setup(N, PFFFT_REAL);

                var* fft = (float*) pffft_aligned_malloc
                (
                    sizeof(float) * 2 * N
                );

                var* result = (float*) pffft_aligned_malloc(sizeof(float) * N);

                if (pffft_simd_size() > 1)
                {
                    printf("pffft_vec, %7.2f, kb, ", 0.0f);
                }
                else
                {
                    printf("pffft_c  , %7.2f, kb, ", 0.0f);
                }

                // -------------------------------------------------------------

                array<chrono::milliseconds, runs> values;

                {
                    pffft_transform
                    (
                          plan_fft
                        , &speed_data[0]
                        , fft
                        , NULL
                        , PFFFT_FORWARD
                    );

                    pffft_transform
                    (
                          plan_fft
                        , fft
                        , result
                        , NULL
                        , PFFFT_BACKWARD
                    );

                    quick_test_64(result, N);
                }

                for (unsigned sample = 0; sample < runs; sample++)
                {
                    let start = chrono::high_resolution_clock::now();

                    for (unsigned offset = 0 ; offset < (size - N); offset+=32)
                    {
                        pffft_transform
                        (
                              plan_fft
                            , &speed_data[offset]
                            , fft
                            , NULL
                            , PFFFT_FORWARD
                        );

                        pffft_transform
                        (
                              plan_fft
                            , fft
                            , result
                            , NULL
                            , PFFFT_BACKWARD
                        );
                    }

                    let end = chrono::high_resolution_clock::now();
                    let delta = end - start;
                    values[sample] =
                        chrono::duration_cast<chrono::milliseconds>(delta);

                    printf(".");
                    fflush(stdout);
                }

                sort(values.begin(), values.end());

                float microseconds_per_fft_multiplier =
                    1000.0f / ((size - N) / 32);

                printf
                (
                      ", %5d, %6zd, ms, %7.2f, microseconds per fft\n"
                    , N
                    , (size_t) values[runs / 2].count()
                    , values[runs / 2].count() * microseconds_per_fft_multiplier
                );

                pffft_destroy_setup(plan_fft);
                pffft_aligned_free(fft);
                pffft_aligned_free(result);

                fflush(stdout);
            }
        }

#ifdef TEST_WITH_FFTW3
        {
            for (let N : test_N)
            {
                var* result = (float*) fftwf_malloc(sizeof(float) * N);
                var* fft    = (float*) fftwf_malloc(sizeof(float) * N);
                var* copy   = (float*) fftwf_malloc(sizeof(float) * N);


                let plan_fft = fftwf_plan_dft_r2c_1d
                (
                      N
                    , copy
                    , (fftwf_complex*) fft
                    , FFTW_MEASURE
                );

                let plan_ifft = fftwf_plan_dft_c2r_1d
                (
                      N
                    , (fftwf_complex*) fft
                    , result
                    , FFTW_MEASURE
                );                

                // creating the plan stomps on the input array.
                memcpy(copy, speed_data.data(), N * sizeof(float));

                printf("fftw3_fft, %7.2f, kb, ", 0.0f);

                // -------------------------------------------------------------

                array<chrono::milliseconds, runs> values;

                {
                    fftwf_execute(plan_fft);
                    fftwf_execute(plan_ifft);

                    quick_test_64(result, N);
                }

                for (unsigned sample = 0; sample < runs; sample++)
                {
                    let start = chrono::high_resolution_clock::now();

                    for (unsigned offset = 0 ; offset < (size - N); offset+=32)
                    {
                        fftwf_execute_dft_r2c
                        (
                              plan_fft
                            , &speed_data[offset]
                            , (fftwf_complex*) fft
                        );

                        fftwf_execute_dft_c2r
                        (
                              plan_fft
                            , (fftwf_complex*) fft
                            , result
                        );
                    }

                    let end = chrono::high_resolution_clock::now();
                    let delta = end - start;
                    values[sample] =
                        chrono::duration_cast<chrono::milliseconds>(delta);

                    printf(".");
                    fflush(stdout);
                }

                sort(values.begin(), values.end());

                float microseconds_per_fft_multiplier =
                    1000.0f / ((size - N) / 32);

                printf
                (
                      ", %5d, %6zd, ms, %7.2f, microseconds per fft\n"
                    , N
                    , values[runs / 2].count()
                    , values[runs / 2].count() * microseconds_per_fft_multiplier
                );

                fftwf_destroy_plan(plan_fft);
                fftwf_destroy_plan(plan_ifft);
                fftwf_free(fft);
                fftwf_free(result);

                fflush(stdout);
            }
        }
#endif
    }

    printf("Done\n");

    fflush(stdout);

    return 0;
}
