##############################################################
########################## TO DO #############################
##############################################################
# make checks for case with quantum numbers
# make checks for general TTN case
# implement global Krylov method 
##############################################################


"""
expand subspace (2-site like) in a sweep 
"""
function subspace_expansion_sweep!(
  ψ::AbstractTTN{V}, PH::AbstractProjTTN{V}; maxdim, cutoff, atol=1e-10, kwargs...
) where {V}

  order = TDVPOrder(2, Base.Forward)
  direction = directions(order)
  HGraph = underlying_graph(PH)
  startPos = default_root_vertex(HGraph)

  if !isortho(ψ) || ortho_center(ψ) != startPos
    ψ = orthogonalize(ψ, startPos)
  end

  nsite = 2
  PH = set_nsite(PH, nsite)
  PH = position(PH, ψ, [startPos])

  ##both forward and backward directions
  for dir in direction
    ##iterate through the tree
    for sweep_step in two_site_sweep(dir, HGraph, startPos, true; state=ψ)
      # isa(pos(sweep_step), AbstractEdge) && continue
      length(pos(sweep_step)) == 1 && continue

      b = pos(sweep_step)
      ha = time_direction(sweep_step)

      ##TODO: figure out whether these calls should be here or inside subspace expansion, currently we do both?
      ψ = orthogonalize(ψ, b[1])
      PH = position(PH, ψ, [b[1]])

      subspace_expansion!(
        ψ, PH, b; maxdim, cutoff=cutoff, atol=atol, kwargs...
      )
    end
  end
  return ψ
end

function _subspace_expand_core(
  centerwf::Vector{ITensor}, env, NL, NR; maxdim, cutoff, kwargs...
)
  ϕ = ITensor(1.0)
  for atensor in centerwf
    ϕ *= atensor
  end
  return _subspace_expand_core(ϕ, env, NL, NR; maxdim, cutoff, kwargs...)
end

function _subspace_expand_core(ϕ::ITensor, env, NL, NR; maxdim, cutoff, kwargs...)
  ϕH = noprime(env(ϕ))   #add noprime?
  ϕH = NL * ϕH * NR
  if norm(ϕH) == 0.0
    return false, false, false, false
  end

  U, S, V = svd(
    ϕH,
    commoninds(ϕH, NL);
    maxdim,
    cutoff,
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
    kwargs...,
  )

  @assert dim(commonind(U, S)) <= maxdim

  NL *= dag(U)
  NR *= dag(V)
  return NL, S, NR, true
end

function subspace_expansion!(
  ψ::ITensorNetworks.AbstractTTN{Vert},
  PH,
  b::Vector{Vert};
  bondtensor=nothing,
  maxdim,
  cutoff,
  atol=1e-8,
  kwargs...,
 ) where {Vert}
  """
  expands in nullspace of site-tensors b
  """
  # atol refers to the tolerance in nullspace determination (for finite MPS can probably be set rather small)
  # cutoff refers to largest singular value of gradient (acceleration of population gain of expansion vectors) to keep
  # this should be related to the cutoff in two-site TDVP: \Delta_rho_i = 0.5 * lambda_y * tau **2 
  # note that in the initial SVD there is another cutoff parameter `cutoff_compress`, that we set to roughly machine precision for the time being
  # (allows to move bond-dimension between different partial QN sectors, if no such truncation occurs distribution of bond-dimensions
  # between different QNs locally is static once bond dimension saturates maxdim.)
  
  n1, n2 = b
  cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)

  # move orthogonality center to bond
  # and check whether there are vanishing contributions to the wavefunctions
  # truncate accordingly
  # at the cost of unitarity but introducing flexibility of redistributing bond dimensions among QN sectors)
  U, S, V = svd(
    ψ[n1], uniqueinds(ψ[n1], ψ[n2]); maxdim=maxdim, cutoff=cutoff_compress, kwargs...
  )
  ϕ_1 = U
  ϕ_2 = ψ[n2] * V
  old_linkdim = dim(commonind(U, S))
  bondtensor = S

  # don't expand if we are already at maxdim
  if old_linkdim >= maxdim
    # println("not expanding")
    return nothing
  end
  PH = position(PH, ψ, [n1,n2])

  # orthogonalize(ψ,n1)
  linkind_l = commonind(ϕ_1, bondtensor)
  linkind_r = commonind(ϕ_2, bondtensor)

  # compute nullspace to the left and right 
  NL = nullspace(ϕ_1, linkind_l; atol=atol)
  NR = nullspace(ϕ_2, linkind_r; atol=atol)

  # if nullspace is empty (happen's for product states with QNs)
  if norm(NL) == 0.0 || norm(NR) == 0.0
    return bondtensor
  end

  # form 2-site wavefunction
  ϕ = ϕ_1 * bondtensor * ϕ_2

  # get subspace expansion
  newL, S, newR, success = _subspace_expand_core(
    ϕ, PH, NL, NR, ; maxdim=maxdim - old_linkdim, cutoff, kwargs...
  )

  # success || println("Subspace expansion not successful. This may indicate that 2-site TDVP also fails for the given state and Hamiltonian.")
  success || return nothing

  # expand current site tensors
  ALⁿ¹, newl = ITensors.directsum(
    ϕ_1 => uniqueinds(ϕ_1, newL), dag(newL) => uniqueinds(newL, ϕ_1); tags=(tags(commonind(ψ[n1],ψ[n2])),)
  )
  ARⁿ², newr = ITensors.directsum(
    ϕ_2 => uniqueinds(ϕ_2, newR), dag(newR) => uniqueinds(newR, ϕ_2); tags=(tags(commonind(ψ[n1],ψ[n2])),)
  )

  # Some checks
  @assert (dim(commonind(newL, S)) + old_linkdim) <= maxdim
  @assert dim(commonind(newL, S)) == dim(commonind(newR, S))
  @assert(dim(uniqueind(ϕ_1, newL)) + dim(uniqueind(newL, ϕ_1)) == dim(newl))
  @assert(dim(uniqueind(ϕ_2, newR)) + dim(uniqueind(newR, ϕ_2)) == dim(newr))
  @assert (old_linkdim + dim(commonind(newL, S))) <= maxdim
  @assert (old_linkdim + dim(commonind(newR, S))) <= maxdim
  @assert dim(newl) <= maxdim
  @assert dim(newr) <= maxdim

  # zero-pad bond-tensor (the orthogonality center)
  C = ITensor(dag(newl)..., dag(newr)...)
  ψC = bondtensor
  for I in eachindex(ψC)
    v = ψC[I]
    if !iszero(v)
      C[I] = ψC[I]
    end
  end

  ψ[n2] = ARⁿ²
  ψ[n1] = noprime(ALⁿ¹ * C)

  return nothing
end
