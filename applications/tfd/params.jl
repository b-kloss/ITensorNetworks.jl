module params

export outf, tdvp_cutoff, tdvp_dt,tdvp_order, chi, log_dt, final_time
export omega,t, gamma, N, boson_dim, temperature
export threading, BLAS_nthreads

const    outf="data.h5"
const    tdvp_cutoff=1e-6
const	 chi = 256
const    tdvp_dt=0.1
const    tdvp_order = 2  #2 or 4(4 doesn't seem to work as well as it should)
const    log_dt=0.1
const    final_time = 1.0
const    omega=1.0
const    t=1.0
const    gamma=sqrt(2.0)
const    threading="hybrid"
const 	 BLAS_nthreads=1
const    N=2
const    boson_dim=21
const    temperature=0.4

end
