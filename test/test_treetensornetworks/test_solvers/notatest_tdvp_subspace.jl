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

auto_fermion_enabled = ITensors.using_auto_fermion()
ITensors.enable_auto_fermion()
@testset "MPS TDVP" begin
  @testset "exact vs 2s" begin
    ITensors.enable_auto_fermion()
    Random.seed!(1234)
    cutoff = 1e-14
    N = 20
    D = 64
    t = 1.0
    tp = 0.0
    dt=0.05
    s = siteinds("Fermion", N,conserve_qns=true)
    os = ITensorNetworks.tight_binding(N; t, tp)
    H = mpo(os, s)
    hmat = hopping_operator(os)
    # get the exact result
    tf=1.0
    taus = LinRange(0, tf, 2)
    init = I(N)[:, 1:(div(N, 2))] #domain wall initial condition
    states = i -> i<=div(N,2) ? "Occ" : "Emp"
    init = conj(init) * transpose(init)
    res = zeros(N, length(taus))
    for (i, tau) in enumerate(taus)
      res[:, i] = real.(diag(propagate_noninteracting(hmat, init, tau))) #densities only
    end

    psi=mps(s;states)
    psi0=deepcopy(psi)
    psif = tdvp(H, -1im*tf, psi; order=2,time_step= -1im*dt, cutoff, nsites=2,outputlevel=1)
    @test inner(psi0,psi) ≈ 1 atol=1e-12  # just making sure that tdvp is out-of-place
    mag_2s=expect("N",psif)
    @show maximum(abs.(real.([mag_2s[v] for v in vertices(psif)]) .- res[:,end]))
    @test all(i->i<5e-3,abs.(real.([mag_2s[v] for v in vertices(psif)]) .- res[:,end]))
  end
  
  @testset "exact vs 1s + local subspace" begin
    ITensors.enable_auto_fermion()
    Random.seed!(1234)
    cutoff = 1e-14
    N = 20
    D = 128
    t = 1.0
    tp = 0.0
    dt=0.05
    s = siteinds("Fermion", N,conserve_qns=true)
    os = ITensorNetworks.tight_binding(N; t, tp)
    H = mpo(os, s)
    hmat = hopping_operator(os)
    # get the exact result
    tf=1.0
    taus = LinRange(0, tf, 2)
    init = I(N)[:, 1:(div(N, 2))] #domain wall initial condition
    states = i -> i<=div(N,2) ? "Occ" : "Emp"
    init = conj(init) * transpose(init)
    res = zeros(N, length(taus))
    for (i, tau) in enumerate(taus)
      res[:, i] = real.(diag(propagate_noninteracting(hmat, init, tau))) #densities only
    end

    psi=mps(s;states)
    psi0=deepcopy(psi)
    tdvp_kwargs = (time_step = -im*dt, reverse_step=true, order=2, normalize=true, maxdim=D, cutoff=cutoff, outputlevel=1,
    updater_kwargs=(;expand_kwargs=(;cutoff=cutoff/dt,svd_func_expand=ITensorNetworks._svd_solve_normal),exponentiate_kwargs=(;)))#,exponentiate_kwargs=(;tol=1e-8)))
    psife = tdvp(ITensorNetworks.local_expand_and_exponentiate_updater,H, -1im*tf, psi; nsites=1, tdvp_kwargs...)
    @test inner(psi0,psi) ≈ 1 atol=1e-12
    mag_exp=expect("N",psife)
    @show maximum(abs.(real.([mag_exp[v] for v in vertices(psife)]) .- res[:,end]))
    @test all(i->i<5e-3,abs.(real.([mag_exp[v] for v in vertices(psife)]) .- res[:,end]))

    #test whether rsvd works for fermions
    tdvp_kwargs = (time_step = -im*dt, reverse_step=true, order=2, normalize=true, maxdim=D, cutoff=cutoff, outputlevel=1,
    updater_kwargs=(;expand_kwargs=(;cutoff=cutoff/dt,svd_func_expand=ITensorNetworks.rsvd_iterative),exponentiate_kwargs=(;)))#,exponentiate_kwargs=(;tol=1e-8)))
    @test_broken tdvp(ITensorNetworks.local_expand_and_exponentiate_updater,H, -1im*tf, psi; nsites=1, tdvp_kwargs...)

  end
end

if !auto_fermion_enabled
  ITensors.disable_auto_fermion()
end
nothing
