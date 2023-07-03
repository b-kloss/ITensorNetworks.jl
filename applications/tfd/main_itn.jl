import MPI
MPI.Init_thread(MPI.THREAD_FUNNELED)
using LinearAlgebra,MKL
using ITensors
using ITensorNetworks
using HDF5
using TickTock
using Observers
#using ConfParser
parameter_file=ARGS[1]
include(parameter_file)
using Main.params
#MPI.Init()
comm = MPI.COMM_WORLD

#threading=Main.params.threading
#outf=Main.params.outf
#tdvp_dt=Main.params.tdvp_dt
#log_dt=Main.params.log_dt
#final_time=Main.params.final_time
#BLAS_nthreads=Main.params.BLAS_nthreads
#gamma=Main.params.gamma
#N=params.N
#N=copy(n)
#boson_dim=params.boson_dim
#t=params.t
#omega=params.omega

include("./boson.jl")
include("./model.jl")
include("./observer.jl")
include("./measure.jl")

function log(fname, oname, time, data)
  fid=h5open(fname,"r+")
  write(fid,oname*"/t"*string(time),data)
  close(fid)
  return nothing
end


function set_threading(threading)
    t=Threads.nthreads()
    if threading=="BLAS"
        BLAS.set_num_threads(t)
        ITensors.Strided.disable_threads()
        ITensors.disable_threaded_blocksparse()
    elseif threading=="Strided"
        BLAS.set_num_threads(1)
        ITensors.Strided.enable_threads()
        ITensors.disable_threaded_blocksparse()
    elseif threading=="blocksparse"
        BLAS.set_num_threads(1)
        ITensors.Strided.disable_threads()
        ITensors.enable_threaded_blocksparse()
    elseif threading=="hybrid"
    	BLAS.set_num_threads(BLAS_nthreads)
	ITensors.enable_threaded_blocksparse()
	ITensors.Strided.enable_threads()
    end
  end
let
#outf="data.h5"
use_MPI = MPI.Comm_size==1 ? false : true
set_threading(threading)
@show N,omega,boson_dim

lambda=gamma^2/(2*t*omega)
alpha=gamma
total_Nsteps=convert(Int, ceil(abs( final_time / log_dt)))
if temperature>0.0
  stride=3
  finitetemp=true
else
  stride=2
  finitetemp=false
end


root = div(N,2)
if MPI.Comm_rank(comm) == root
    print(" Running on $(MPI.Comm_size(comm)) processes\n")

end
if use_MPI
    @assert MPI.Comm_size(comm)==N
end
rank=MPI.Comm_rank(comm)
MPIsize=MPI.Comm_size(comm)


if rank==root
  outfid=h5open(outf,"w")
  close(outfid)
end

tdvp_kwargs = (time_step = -im*tdvp_dt, reverse_step=true, normalize=true, maxdim=Main.params.chi, cutoff=tdvp_cutoff, outputlevel=1, cutoff_compress=1e-14)

phys_bos = siteinds("MyBoson", N,dim=boson_dim,conserve_qns=true,conserve_number=false,)
ancs_bos = siteinds("MyBoson", N,dim=boson_dim, conserve_qns=true,conserve_number=false)
els = siteinds("Fermion",N,conserve_qns=true)
if finitetemp
  ancs_bos = siteinds("MyBoson", N,dim=boson_dim, conserve_qns=true,conserve_number=false)
  ancs_bos = addtags(ancs_bos,",ancilla")
end
#sites=Vector{Index{Vector{Pair{QN, Int64}}}}()
sites=Vector{Index}()
if finitetemp
  for (x,y,z) in zip(phys_bos,els,ancs_bos)
      append!(sites,(x,y,z))
  end
else
  for (x,y) in zip(phys_bos,els)
    append!(sites,(x,y))
  end
end
#new_log_dt=tdvp_dt

function get_mytdvpobs()
  obsPH = Observer("PH" => return_PH)
end

function tdvp_init(psi::TTN,H)
  PHobs=get_mytdvpobs()
  psi=tdvp(H,-im*log_dt,psi;tdvp_kwargs...,
  expander_backend="full", 
  svd_backend="svd",
  nsite=1,
  expander_cache=expander_cache,
  (observer!)=PHobs,)
  PH=last(PHobs["PH"][2])
  @show typeof(PH)
  return psi,PH
end

function tdvp_step!(psi::TTN,PH)
  PHobs=get_mytdvpobs()
  psi=tdvp(PH,-im*log_dt,psi;tdvp_kwargs...,
  expander_backend="full", 
  svd_backend="svd",
  nsite=1,
  expander_cache=expander_cache,
  (observer!)=PHobs)
  PH=PHobs[end, "PH"]
  return psi,PH
end

function propagate!(ψ,PH,propfunc)
  #ldims=linkdims(ψ)
  ψ,PH=propfunc(ψ,PH)
  return ψ,PH
end

if use_MPI
    if finitetemp
        opsum=tfd_holstein(N;omega=omega,t=t,alpha=alpha, T=temperature, order=["phys_bos","el","anc_bos"])
        states = [n == rank+1 ? [1,"Occ",1] : [1,"Emp",1] for n=1:N]
    else
        opsum=tfd_holstein(N;omega=omega,t=t,alpha=alpha, T=0.0, order=["phys_bos","el"])
        states = [n == rank+1 ? [1,"Occ"] : [1,"Emp"] for n=1:N]
    end
else
    if finitetemp
        opsum=tfd_holstein(N;omega=omega,t=t,alpha=alpha, T=temperature, order=["phys_bos","el","anc_bos"])
        states = [n == div(N,2)+1 ? [1,"Occ",1] : [1,"Emp",1] for n=1:N]
    else
        opsum=tfd_holstein(N;omega=omega,t=t,alpha=alpha, T=0.0, order=["phys_bos","el"])
        states = [n == div(N,2)+1 ? [1,"Occ"] : [1,"Emp"] for n=1:N]
    end
end

states=reduce(vcat,states)
@show states
@show typeof(sites[1])
ψ = MPS(ComplexF64,sites, states)
ψtn = TTN([ψ[i] for i in eachindex(ψ)])
sites_tn=siteinds(ψtn)
Nphys_sites=div(length(ψ),stride)
H = MPO(opsum,sites)
Htn = TTN(opsum,sites_tn)
expander_cache=Any[]
tdvp_kwargs = (time_step = -im*tdvp_dt, reverse_step=true, normalize=true, maxdim=Main.params.chi, cutoff=tdvp_cutoff, outputlevel=1, cutoff_compress=1e-12)

PH=ProjTTN(Htn)
ψ0tn=deepcopy(ψtn)
ϕtn=nothing
##compute expectation values at t=0
if rank==root
    println("t=0 observables")
end
#ϕ0 = nothing
#ϕ0 = MPI.bcast(ψ0,root,comm)
if use_MPI
    ϕtn = MPI.bcast(ψtn,root,comm)
else
    ϕtn = copy(ψtn)
end
if use_MPI
    if finitetemp
        allgf,allgf2 = measure_corr(ψtn,ϕtn,ψ0tn,stride=3)
    else
        allgf,allgf2 = measure_corr(ψtn,ϕtn,ψ0tn,stride=2)
    end
else
    if finitetemp
        allgf = measure_autocorr(ψtn)
    else
        allgf = measure_autocorr(ψtn)
    end
end
    @show allgf
if use_MPI
    E = inner(ψ',H,ψ)
    #allgf=MPI.Gather(gf, root,comm)
    allEs=MPI.Gather(E, root,comm)
end
if rank==root
    #@show allgf
    #log(outf,"E",0.0,allEs)
   # @show Matrix(reduce(hcat,real(allgf))')
    if typeof(allgf)<:Number
        log(outf,"gf_real",0.0,real(allgf))
        log(outf,"gf_imag",0.0,imag(allgf))
    else
        log(outf,"gf_real",0.0,Matrix(reduce(hcat,real(allgf))'))
        log(outf,"gf2_real",0.0,Matrix(reduce(hcat,real(allgf2))'))
        log(outf,"gf_imag",0.0,Matrix(reduce(hcat,imag(allgf))'))
        log(outf,"gf2_imag",0.0,Matrix(reduce(hcat,imag(allgf2))'))
    end
    
    
end
if rank==root
    println("starting prop")
end


for i in range(1,total_Nsteps*5)
  if rank==root
    @time ψtn,PH=propagate!(ψtn,PH,tdvp_step!)
    #println(maxlinkdim(ψ)," ",minimum(linkdims(ψ)))
  else
    ψtn,PH=propagate!(ψtn,PH,tdvp_step!)
  end
  if rank==root
    tick()
  end
  if use_MPI
    if finitetemp
        allgf,allgf2 = measure_corr(ψtn,ϕtn,ψ0tn,stride=3)
    else
        allgf,allgf2 = measure_corr(ψtn,ϕtn,ψ0tn,stride=2)
    end
  else
    if finitetemp
        allgf = measure_autocorr(ψtn)
    else
        allgf = measure_autocorr(ψtn)
    end
    @show allgf
  end

  if rank==root
    if typeof(allgf)<:Number
        log(outf,"gf_real",log_dt*i*2,real(allgf))
        log(outf,"gf_imag",log_dt*i*2,imag(allgf))
    else
        #log(outf,"E",log_dt*i,allEs)
        log(outf,"gf_real",log_dt*i*2,Matrix(reduce(hcat,real(allgf))'))
        log(outf,"gf2_real",log_dt*i,Matrix(reduce(hcat,real(allgf2))'))
        log(outf,"gf_imag",log_dt*i*2,Matrix(reduce(hcat,imag(allgf))'))
        log(outf,"gf2_imag",log_dt*i,Matrix(reduce(hcat,imag(allgf2))'))
    end
  end
  MPI.Barrier(comm)
  if rank==root
    tock()
    println("computing gfs and logging took overall")
  end
end
#PH=ProjMPO(H)
end


  
