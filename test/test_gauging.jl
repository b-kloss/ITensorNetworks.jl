@eval module $(gensym())
using Compat: Compat
using ITensorNetworks:
  BeliefPropagationCache,
  ITensorNetwork,
  VidalITensorNetwork,
  contract_inner,
  gauge_error,
  messages,
  random_tensornetwork,
  siteinds,
  update
using ITensors: diagITensor, inds
using ITensors.NDTensors: vector
using LinearAlgebra: diag
using NamedGraphs: named_grid
using Random: Random
using Test: @test, @testset

@testset "gauging" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 6

  Random.seed!(5467)
  ψ = random_tensornetwork(s; link_space=χ)

  # Move directly to vidal gauge
  ψ_vidal = VidalITensorNetwork(ψ)
  @test gauge_error(ψ_vidal) < 1e-5

  # Move to symmetric gauge
  cache_ref = Ref{BeliefPropagationCache}()
  ψ_symm = ITensorNetwork(ψ_vidal; (cache!)=cache_ref)
  bp_cache = cache_ref[]

  # Test we just did a gauge transform and didn't change the overall network
  @test contract_inner(ψ_symm, ψ) /
        sqrt(contract_inner(ψ_symm, ψ_symm) * contract_inner(ψ, ψ)) ≈ 1.0

  #Test all message tensors are approximately diagonal even when we keep running BP
  bp_cache = update(bp_cache; maxiter=20)
  for m_e in values(messages(bp_cache))
    @test diagITensor(vector(diag(only(m_e))), inds(only(m_e))) ≈ only(m_e) atol = 1e-8
  end
end
end
