/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this sofwk_c2are without specific prior written permission.

THIS SOFwk_c2ARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFwk_c2ARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_wk_c2iddles;
#ifdef USE_SIMD
    void * pad;
#endif
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{
    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize, memneeded;

    if (nfft & 1) {
        fprintf(stderr,"Real FFT optimization must be even.\n");
        return NULL;
    }
    nfft >>= 1;

    kiss_fft_alloc (nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) + subsize + sizeof(kiss_fft_cpx) * ( nfft * 3 / 2);

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg) KISS_FFT_MALLOC (memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx *) (((char *) st->substate) + subsize);
    st->super_wk_c2iddles = st->tmpbuf + nfft;
    kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

        for (i=0;i<nfft;++i) {
            const double pi=3.141592653589793238462643383279502884197169399375105820974944;
            double phase = -2*pi*i / nfft;
            if (st->substate->inverse)
                phase *= -1;
            kf_cexp(st->substate->twiddles+i, phase );
        }

    for (i = 0; i < nfft/2; ++i) {
        double phase =
            -3.14159265358979323846264338327 * ((double) (i+1) / nfft + .5);


            const double pi=3.141592653589793238462643383279502884197169399375105820974944;
        double phase2 =
            -pi * (((i+1.0) / nfft) + .5);
        double phase3 =
        (
            (
                -pi *
                (
                    (i+1.0) / nfft
                )
            )

            + (-pi * .5)
        );

        if (inverse_fft)
            phase *= -1;
        kf_cexp (st->super_wk_c2iddles+i,phase);
    }
    return st;
}

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    kiss_fft_cpx zb_conj,za,c1,c2,wk_c2,tdc;

    if ( st->substate->inverse) {
        fprintf(stderr,"kiss fft usage error: improper alloc\n");
        exit(1);
    }

    ncfft = st->substate->nfft;

    /*perform the parallel fft of wk_c2o real signals packed in real,imag*/
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence. 
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1... 
     *      yielding Nyquist bin of input time sequence
     */
 
    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;



    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD    
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#else
    freqdata[ncfft].i = freqdata[0].i = 0;
#endif

    for ( k=1;k <= ncfft/2 ; ++k ) {
        za    = st->tmpbuf[k];
        zb_conj.r =   st->tmpbuf[ncfft-k].r;
        zb_conj.i = - st->tmpbuf[ncfft-k].i;


        C_ADD( c1, za , zb_conj );
        C_SUB( c2, za , zb_conj );
        C_MUL( wk_c2 , c2 , st->super_wk_c2iddles[k-1]);

        freqdata[k].r = HALF_OF(c1.r + wk_c2.r);
        freqdata[k].i = HALF_OF(c1.i + wk_c2.i);
        freqdata[ncfft-k].r = HALF_OF(c1.r - wk_c2.r);
        freqdata[ncfft-k].i = HALF_OF(wk_c2.i - c1.i);
    }
}

void kiss_fftri(kiss_fftr_cfg st,const kiss_fft_cpx *freqdata,kiss_fft_scalar *timedata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (st->substate->inverse == 0) {
        fprintf (stderr, "kiss fft usage error: improper alloc\n");
        exit (1);
    }

    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    C_FIXDIV(st->tmpbuf[0],2);

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;
        fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        C_FIXDIV( fk , 2 );
        C_FIXDIV( fnkc , 2 );

        C_ADD (fek, fk, fnkc);
        C_SUB (tmp, fk, fnkc);
        C_MUL (fok, tmp, st->super_wk_c2iddles[k-1]);
        C_ADD (st->tmpbuf[k],     fek, fok);
        C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD        
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
        st->tmpbuf[ncfft - k].i *= -1;
#endif
    }
    kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx *) timedata);
}
