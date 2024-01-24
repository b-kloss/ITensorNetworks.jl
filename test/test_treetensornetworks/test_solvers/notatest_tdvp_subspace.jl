using ITensors
using ITensorNetworks
using ITensorNetworks: exponentiate_updater,_svd_solve_normal
using KrylovKit: exponentiate
using Observers
using Random
using Test
using ITensorGaussianMPS: hopping_operator
using LinearAlgebra: exp, I

"""
Propagate a Gaussian density matrix `c` with a noninteracting Hamiltonian `h` up to time `tau`.
"""
function propagate_noninteracting(h::AbstractMatrix, c::AbstractMatrix, tau::Number)
  U = exp(-im * tau * h)
  return U * c * conj(transpose(U))
end

@testset "MPS TDVP" begin
  @testset "2s vs 1s + local subspace" begin
    ITensors.enable_auto_fermion()
    Random.seed!(1234)
    cutoff = 1e-14
    N = 20
    D = 64
    t = 1.0
    tp = 0.0
    s = siteinds("Fermion", N,conserve_qns=true)
    os = ITensorNetworks.tight_binding(N; t, tp)
    H = mpo(os, s)
    hmat = hopping_operator(os)
    # get the exact result
    tf=1.0
    taus = LinRange(0, tf, 2)
    #init=I(N)[:,StatsBase.sample(1:N, div(N,2); replace=false)]
    init = I(N)[:, 1:(div(N, 2))] #domain wall initial condition
    states = i -> i<=div(N,2) ? "Occ" : "Emp"
    init = conj(init) * transpose(init)
    #res=[]
    res = zeros(N, length(taus))
    for (i, tau) in enumerate(taus)
      res[:, i] = real.(diag(propagate_noninteracting(hmat, init, tau))) #densities only
      #plot!(1:N, real.(diag(last(res))))
    end
    psi=mps(s;states)
    psi0=deepcopy(psi)
    dt=0.025
    @show siteinds(psi)
    #tdvp_kwargs = (time_step = -im*dt, reverse_step=true, order=4, normalize=true, maxdim=D, cutoff=cutoff, outputlevel=1,
    #updater_kwargs=(;expand_kwargs=(;cutoff=cutoff/dt,svd_func_expand=_svd_solve_normal),exponentiate_kwargs=(;)))#,exponentiate_kwargs=(;tol=1e-8)))
    success=false
    psif=nothing
    while !success
      try
        psif = tdvp(H, -1im*tf, psi; order=2,time_step= -1im*dt, cutoff, nsites=2,outputlevel=1)
        success=true
      catch
        println("trying again")
      end
    end
    
    @test inner(psi0,psi) ≈ 1 atol=1e-12
    #@show maxlinkdim(psife)
    mag_2s=expect("N",psif)
    #mag_exp=expect("N",psif)
   # @test real.([mag_2s[v] for v in vertices(psif)]) ≈ res[:,end] atol=1e-3
    #@test real.([mag_2s[v] for v in vertices(psif)]) ≈ res[:,end] atol=1e-3
    @test all(i->i<5e-3,abs.(real.([mag_2s[v] for v in vertices(psif)]) .- res[:,end]))
  end
  @testset "2s vs 1s + local subspace" begin
    ITensors.enable_auto_fermion()
    Random.seed!(1234)
    cutoff = 1e-14
    N = 20
    D = 128
    t = 1.0
    tp = 0.0
    s = siteinds("Fermion", N,conserve_qns=true)
    os = ITensorNetworks.tight_binding(N; t, tp)
    H = mpo(os, s)
    hmat = hopping_operator(os)
    # get the exact result
    tf=1.0
    taus = LinRange(0, tf, 2)
    #init=I(N)[:,StatsBase.sample(1:N, div(N,2); replace=false)]
    init = I(N)[:, 1:(div(N, 2))] #domain wall initial condition
    states = i -> i<=div(N,2) ? "Occ" : "Emp"
    init = conj(init) * transpose(init)
    #res=[]
    res = zeros(N, length(taus))
    for (i, tau) in enumerate(taus)
      res[:, i] = real.(diag(propagate_noninteracting(hmat, init, tau))) #densities only
      #plot!(1:N, real.(diag(last(res))))
    end
    psi=mps(s;states)
    psi0=deepcopy(psi)
    dt=0.025
    @show siteinds(psi)
    tdvp_kwargs = (time_step = -im*dt, reverse_step=true, order=2, normalize=true, maxdim=D, cutoff=cutoff, outputlevel=1,
    updater_kwargs=(;expand_kwargs=(;cutoff=cutoff/dt,svd_func_expand=_svd_solve_normal),exponentiate_kwargs=(;)))#,exponentiate_kwargs=(;tol=1e-8)))
    success=false
    psife=nothing
    while !success
      try
        psife = tdvp(ITensorNetworks.local_expand_and_exponentiate_updater,H, -1im*tf, psi; nsites=1, tdvp_kwargs...)
        success=true
      catch
        println("trying again")
      end
      
    end
    
    
    @test inner(psi0,psi) ≈ 1 atol=1e-12
    #@show maxlinkdim(psife)
    #mag_2s=expect("N",psif)
    mag_exp=expect("N",psife)
   # @test real.([mag_2s[v] for v in vertices(psif)]) ≈ res[:,end] atol=1e-3
    @test all(i->i<5e-3,abs.(real.([mag_exp[v] for v in vertices(psife)]) .- res[:,end]))
  end

end
nothing
