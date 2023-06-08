##############################################################
########################## TO DO #############################
##############################################################
# make checks for case with quantum numbers
#     do this for dmrg routine to compare energies, since expect seems to be broken for QN's
# make checks for general TTN case
# implement global Krylov method 
##############################################################


"""
expand subspace (2-site like) in a sweep 
"""
function subspace_expansion_full_sweep!(
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

      subspace_expansion_full!(
        ψ, PH, b; maxdim, cutoff=cutoff, atol=atol, kwargs...
      )

    end
  end
  return ψ
end

function _subspace_expand_full_core(
  centerwf::Vector{ITensor}, PH, Nn1; maxdim, cutoff, kwargs...
)
  ϕ = reduce(*, centerwf; init = ITensor(1.))
  return _subspace_expand_full_core(ϕ, PH, Nn1; maxdim, cutoff, kwargs...)
end

function _subspace_expand_full_core(ϕ::Vector{ITensor}, PHn1, PHn2, Nn1; maxdim, cutoff, kwargs...)

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

function subspace_expansion_full!(
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
  g = underlying_graph(PH)

  # move orthogonality center to bond
  # and check whether there are vanishing contributions to the wavefunctions
  # truncate accordingly
  # at the cost of unitarity but introducing flexibility of redistributing bond dimensions among QN sectors)
  #
  ### use ITensorNetworks.svd function??
  U, S, V = svd(
    ψ[n1], uniqueinds(ψ[n1], ψ[n2]); maxdim=maxdim, cutoff=cutoff_compress, kwargs...
  )
  ϕ_1 = U
  ϕ_2 = ψ[n2] * V
  bondtensor = S
  old_linkdim = dim(commonind(U, S))
  # bondtensor = S

  # don't expand if we are already at maxdim
  old_linkdim >= maxdim && return nothing
  PH = position(PH, ψ, [n1])

  ## brute-force for now, need a better way to do that
  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn1 = reduce(*, PHn1, init=ϕ_1*bondtensor*PH.H[n1])
  if n2 > n1
    PHn2 = mapreduce(*, filter(x->x>n2, vertices(ψ)); init=ϕ_2*PH.H[n2]) do v
      return ψ[v]*PH.H[v]
    end
  else
    PHn2 = mapreduce(*, filter(x->x<n2, vertices(ψ)); init=ϕ_2*PH.H[n2]) do v
      return ψ[v]*PH.H[v]
    end
  end


  # orthogonalize(ψ,n1)
  linkind_l = commonind(ϕ_1, S)
  # linkind_r = commonind(ϕ_2, bondtensor)

  # compute nullspace to the left and right 
  Nn1 = nullspace(ϕ_1, linkind_l; atol=atol)
  # NR = nullspace(ϕ_2, linkind_r; atol=atol)

  # if nullspace is empty (happen's for product states with QNs)
  norm(Nn1) == 0.0 && return bondtensor

  # form 2-site wavefunction
  ϕ = [ϕ_1, bondtensor, ϕ_2]

  ϕH = noprime(PHn1)*Nn1*PHn2
  
  # success || println("Subspace expansion not successful. This may indicate that 2-site TDVP also fails for the given state and Hamiltonian.")
  norm(ϕH) == 0. && return nothing

  # get subspace expansion
  
  # newL, S, newR, success = _subspace_expand_full_core(
  #   ϕ, PHn1, PHn2, Nn1; maxdim=maxdim - old_linkdim, cutoff, kwargs...
  # )
  
  U, S, V = svd(
    ϕH,
    commoninds(ϕH, Nn1);
    maxdim,
    cutoff,
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
    kwargs...,
  )

  newL = Nn1*dag(U)

  # expand current site tensors
  ALⁿ¹, newl = ITensors.directsum(
    ϕ_1 => uniqueinds(ϕ_1, newL), dag(newL) => uniqueinds(newL, ϕ_1); tags=(tags(commonind(ψ[n1],ψ[n2])),)
  )
  # ARⁿ², newr = ITensors.directsum(
  #   ϕ_2 => uniqueinds(ϕ_2, newR), dag(newR) => uniqueinds(newR, ϕ_2); tags=(tags(commonind(ψ[n1],ψ[n2])),)
  # )

  # Some checks
  # @assert (dim(commonind(newL, S)) + old_linkdim) <= maxdim
  # @assert dim(commonind(newL, S)) == dim(commonind(newR, S))
  # @assert(dim(uniqueind(ϕ_1, newL)) + dim(uniqueind(newL, ϕ_1)) == dim(newl))
  # @assert(dim(uniqueind(ϕ_2, newR)) + dim(uniqueind(newR, ϕ_2)) == dim(newr))
  # @assert (old_linkdim + dim(commonind(newL, S))) <= maxdim
  # @assert (old_linkdim + dim(commonind(newR, S))) <= maxdim
  # @assert dim(newl) <= maxdim
  # @assert dim(newr) <= maxdim

  # zero-pad bond-tensor (the orthogonality center)
  C = ITensor(dag(newl)..., commonind(bondtensor, ϕ_2))
  ψC = bondtensor
  for I in eachindex(ψC)
    v = ψC[I]
    if !iszero(v)
      C[I] = ψC[I]
    end
  end
  # @show inds(C)
  # @show inds(ψ[n1])
  # @show inds(ψ[n2])
  # @show inds(ALⁿ¹)

  ψ[n1] = noprime(ALⁿ¹)
  ψ[n2] = ϕ_2 * C
  # @show inds(ψ[n1])
  # @show inds(ψ[n2])

  return nothing
end
