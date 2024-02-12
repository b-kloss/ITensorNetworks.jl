using ITensors
using ITensorNetworks
using ITensorNetworks: rsvd
using Test
using Random

ITensors.enable_auto_fermion()
@testset "RSVD single shot non-fermionic no QNs" begin
  N = 12
  Na = div(N, 2) - 2
  s = siteinds("S=1/2", N; conserve_qns=false)
  linds = s[1:Na]
  rinds = s[(Na + 1):end]
  #target_flux=QN("Sz",4,1)
  #T=randomITensor(target_flux,linds...,rinds...)
  T = randomITensor(linds..., rinds...)
  maxdim = 7
  U, S, V = svd(T, linds...; maxdim)
  Tt = U * S * V
  Tt /= norm(Tt)
  Ur, Sr, Vr = rsvd(Tt, linds, maxdim, div(maxdim, 2))
  @show norm(Ur), norm(Sr), norm(Vr)
  @show norm(Ur * Sr * Vr - Tt)
  @test norm(Ur * Sr * Vr - Tt) ≈ 0.0 atol = 1e-5
end

@testset "RSVD single shot fermionic" begin
  @show ITensors.using_auto_fermion()
  N = 6
  Na = div(N, 2)
  s = siteinds("Fermion", N; conserve_qns=true)
  linds = s[1:Na]
  rinds = s[(Na + 1):end]
  target_flux = QN("Nf", div(N, 2), -1)
  T = randomITensor(target_flux, linds..., rinds...)
  @show norm(T)
  #@show T
  maxdim = 7
  U, S, V = ITensors.svd(T, linds...; maxdim)
  Tt = U * S * V
  @show norm(Tt)
  Tt /= norm(Tt)
  @show norm(Tt)
  Ur, Sr, Vr = rsvd(Tt, linds, maxdim, div(maxdim, 2);)
  @show norm(Ur), norm(Sr), norm(Vr)
  @show norm(U), norm(S), norm(V)
  @show norm(Ur * Sr * Vr - Tt)
  @test norm(Ur * Sr * Vr - Tt) ≈ 0.0 atol = 1e-5
end

@testset "RSVD vector single shot fermionic" begin
  @show ITensors.using_auto_fermion()
  maxdim = 8
  N = 15
  Na = div(N, 3)
  s = siteinds("Fermion", N; conserve_qns=true)
  linds = s[1:Na]
  cinds = s[(Na + 1):(2 * Na)]
  rinds = s[(2 * Na + 1):end]
  target_flux = QN("Nf", div(N, 3) - 1, -1)
  T1 = randomITensor(target_flux, linds..., cinds...)
  uT1, sT1, vT1 = ITensors.svd(T1, linds...; maxdim)
  target_flux = QN("Nf", div(N, 3) - 2, -1)
  T2 = randomITensor(target_flux, dag.(cinds)..., rinds...)
  tmap = [uT1 * sT1, vT1 * T2]
  @show hasqns.(tmap)
  maxdim = 7
  Ur, Sr, Vr = rsvd(tmap, linds, maxdim, div(maxdim, 2))
  Ur, Sr, Vr = rsvd(tmap, linds, maxdim, div(maxdim, 2))
  @show norm(Ur * Sr * Vr - contract(tmap))
  @test norm(Ur * Sr * Vr - contract(tmap)) ≈ 0.0 atol = 1e-5
end

@testset "RSVD ITensor iterative fermionic" begin
  @show ITensors.using_auto_fermion()
  N = 8
  Na = div(N, 2)
  s = siteinds("Fermion", N; conserve_qns=true)
  linds = s[1:Na]
  rinds = s[(Na + 1):end]
  target_flux = QN("Nf", div(N, 2), -1)
  T = randomITensor(target_flux, linds..., rinds...)
  @show norm(T)
  #@show T
  maxdim = 7
  U, S, V = ITensors.svd(T, linds...; maxdim)
  Tt = U * S * V
  old_norm = norm(Tt)
  Tt /= old_norm
  Ur, Sr, Vr = ITensorNetworks.rsvd_iterative(Tt, linds; maxdim)
  @show norm(Ur), norm(Sr), norm(Vr)
  @show norm(U), norm(S) / old_norm, norm(V)
  @show norm(Ur * Sr * Vr - Tt)
  @test norm(Ur * Sr * Vr - Tt) ≈ 0.0 atol = 1e-5
end

@testset "RSVD Vector iterative fermionic" begin
  @show ITensors.using_auto_fermion()
  maxdim = 8
  N = 15
  Na = div(N, 3)
  s = siteinds("Fermion", N; conserve_qns=true)
  linds = s[1:Na]
  cinds = s[(Na + 1):(2 * Na)]
  rinds = s[(2 * Na + 1):end]
  target_flux = QN("Nf", div(N, 3) - 1, -1)
  T1 = randomITensor(target_flux, linds..., cinds...)
  uT1, sT1, vT1 = ITensors.svd(T1, linds...; maxdim)
  target_flux = QN("Nf", div(N, 3) - 2, -1)
  T2 = randomITensor(target_flux, dag.(cinds)..., rinds...)
  tmap = [uT1 * sT1, vT1 * T2]
  @show hasqns.(tmap)
  Ur, Sr, Vr = ITensorNetworks.rsvd_iterative(tmap, linds; maxdim)
  @show norm(Ur), norm(Sr), norm(Vr)
  #@show norm(U), norm(S), norm(V)
  @show norm(Ur * Sr * Vr - contract(tmap))
  @test norm(Ur * Sr * Vr - contract(tmap)) ≈ 0.0 atol = 1e-5
end

#=
    #@show T
    maxdim=7
    U,S,V=ITensors.svd(T,linds...;maxdim)
    Tt=U*S*V
    @show norm(Tt)
    Tt/=norm(Tt)
    @show norm(Tt)
    Ur,Sr,Vr=rsvd(Tt,linds,maxdim,div(maxdim,2);)
    @show norm(Ur), norm(Sr), norm(Vr)
    @show norm(U), norm(S), norm(V)
    @show norm(Ur*Sr*Vr - Tt)
    @test norm(Ur*Sr*Vr - Tt) ≈ 0.0 atol=1e-5
end
=#
nothing

ITensors.disable_auto_fermion()
