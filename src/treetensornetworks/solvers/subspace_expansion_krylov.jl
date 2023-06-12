# atol controls the tolerance cutoff for determining which eigenvectors are in the null
# space of the isometric MPS tensors. Setting to 1e-2 since we only want to keep
# the eigenvectors corresponding to eigenvalues of approximately 1.

###the most general implementation may be passing MPS, lims, and bondtensor (defaulting to the Id-matrix)
###then we can tensorsum both left and right tensor, and just apply the bondtensor to restore input gauge
###ideally we would also be able to apply tensorsum to the bond tensor instead of that loop below
"""
expand subspace (2-site like) in a sweep 
"""
function subspace_expansion_krylov_sweep!(
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

      subspace_expansion_krylov!(
        ψ, PH, b; maxdim, cutoff=cutoff, atol=atol, kwargs...
      )
    end
  end
  return ψ
end

function subspace_expansion_krylov!(
  ψ::ITensorNetworks.AbstractTTN{Vert},
  PH,
  b::Vector{Vert};
  bondtensor=nothing,
  maxdim,
  cutoff,
  atol=1e-8,
  kwargs...,
 ) where {Vert}

  ##this should only work for the case where rlim-llim > 1
  ##not a valid MPS otherwise anyway (since bond matrix not part of MPS unless so defined like in VidalMPS struct)
  n1, n2 = b
  old_linkdim = dim(commonind(ψ[n1], ψ[n2]))
  cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)

  ###move orthogonality center to bond, check whether there are vanishing contributions to the wavefunctions and truncate accordingly
  ###the cutoff should be scaled with timestep, otherwise one runs into problems with non-monotonic error behaviour like in TEBD approaches
  U, S, V = svd(ψ[n1], uniqueinds(ψ[n1], ψ[n2]); maxdim=maxdim, cutoff=1e-12, kwargs...)
  ϕ_1 = U
  ϕ_2 = ψ[n2] * V
  old_linkdim = dim(commonind(U, S))
  bondtensor = S

  ###don't expand if we are already at maxdim
  if old_linkdim >= maxdim
    # println("not expanding")
    return nothing
  end
  PH = position(PH, ψ, [n1,n2])

  #orthogonalize(ψ,n1)
  linkind_l = commonind(ϕ_1, bondtensor)
  linkind_r = commonind(ϕ_2, bondtensor)

  NL = nullspace(ϕ_1, linkind_l; atol=atol)
  NR = nullspace(ϕ_2, linkind_r; atol=atol)
  # @show inds(ψ[1])
  # @show inds(ϕ_1)
  # @show inds(NL)

  ###NOTE: This will fail for rank-1 trees with physical DOFs on leafs
  ###NOTE: one-sided subspace expansion seems to not work well at least for trees according to Lachlan Lindoy
  if norm(NL) == 0.0 || norm(NR) == 0.0
    return bondtensor
  end

  ###form 2site wavefunction
  ϕ = [ϕ_1, bondtensor, ϕ_2]

  ###get subspace expansion
  newL, S, newR, success = _subspace_expand_core_krylov(
    ϕ, PH, NL, NR, b; maxdim=maxdim - old_linkdim, cutoff, kwargs...
  )

  success || return nothing

  ##add expansion direction to current site tensors
  ALⁿ¹, newl = ITensors.directsum(
    ϕ_1 => uniqueinds(ϕ_1, newL)[1], dag(newL) => uniqueinds(newL, ϕ_1)[1]; tags=tags(commonind(ψ[n1],ψ[n2]))
  )
  ARⁿ², newr = ITensors.directsum(
    ϕ_2 => uniqueinds(ϕ_2, newR)[1], dag(newR) => uniqueinds(newR, ϕ_2)[1]; tags=tags(commonind(ψ[n1],ψ[n2]))
  )

  ###TODO remove assertions regarding expansion not exceeding maxdim
  # @assert (dim(commonind(newL, S)) + old_linkdim) <= maxdim
  # @assert dim(commonind(newL, S)) == dim(commonind(newR, S))
  # @assert(dim(uniqueind(ϕ_1, newL)) + dim(uniqueind(newL, ϕ_1)) == dim(newl))
  # @assert(dim(uniqueind(ϕ_2, newR)) + dim(uniqueind(newR, ϕ_2)) == dim(newr))
  # @assert (old_linkdim + dim(commonind(newL, S))) <= maxdim
  # @assert (old_linkdim + dim(commonind(newR, S))) <= maxdim
  # @assert dim(newl) <= maxdim
  # @assert dim(newr) <= maxdim

  ###zero-pad bond-tensor (the orthogonality center)
  C = ITensor(newl, newr)
  ψC = bondtensor
  ### FIXME: the permute below fails, maybe because this already the layout of bondtensor --- in any case it shouldn't fail?
  #ψC = permute(bondtensor, linkind_l, linkind_r)
  for I in eachindex(ψC)
    v = ψC[I]
    if !iszero(v)
      C[I] = ψC[I]
    end
  end

  ###move orthogonality center back to site (should restore input orthogonality limits)
  ψ[n2] = ARⁿ²
  ψ[n1] = noprime(ALⁿ¹ * C)

  return nothing
end

function _subspace_expand_core_krylov(
  centerwf::Vector{ITensor}, PH, NL, NR, b; maxdim, cutoff, kwargs...
)
  (n1,n2) = b
  g = underlying_graph(PH)
  maxdim = min(maxdim, 15)

  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn2 = map(e -> PH.environments[NamedEdge(e)], [v => n2 for v in neighbors(g, n2) if v != n1])
  PHsiteMPOs = map(n -> PH.H[n], b)
  ##NOTE: requires n1+1=n2, otherwise assignment of left and right is of
  L = reduce(*, PHn1, init=centerwf[1])
  L *= PHsiteMPOs[1]
  R = reduce(*, PHn2, init=centerwf[3])
  R *= PHsiteMPOs[2]
  C = centerwf[2]
  R *= C  ##contract C into one of them because otherwise the application is more costly?
  R = noprime(R)
  L = noprime(L)
  outinds = uniqueinds(NL, NL)
  ininds = uniqueinds(NR, NR)
  B2 = ITensorNetworkMaps.ITensorNetworkMap([NR, R, L, NL], ininds, outinds)
  B2dag = adjoint(B2)
  trial = randomITensor(eltype(L), uniqueinds(NL, L))
  # trialadj = randomITensor(eltype(R), uniqueinds(NR, R))   #columnspace of B2, i.e. NL
  #trial=noprime(B2(noprime(B2dag(trial))))
  #vals, lvecs, rvecs, info = svdsolve(trial, maxdim) do (x, flag)
  #  if flag
  #      y = B2dag * copy(x)# y = compute action of adjoint map on x
  #  else
  #      y = B2 * copy(x)# y = compute action of linear map on x
  #  end
  #  return y
  #end
  vals, lvecs, rvecs, info = KrylovKit.svdsolve(
    (x -> noprime(B2 * x), y -> noprime(B2dag * y)), trial, tol=cutoff,  
  )    

  ### cutoff unnecessary values and expand the resulting vectors by a dummy index
  data_cutoff = filter(v->v[1]>cutoff, collect(zip(vals,lvecs,rvecs)))
  (length(data_cutoff) > maxdim) && (data_cutoff = data_cutoff[1:maxdim])

  vals  = [d[1] for d in data_cutoff]
  lvecs = [d[2] for d in data_cutoff]
  rvecs = [d[3] for d in data_cutoff]

  lvecs = map(enumerate(lvecs)) do (i,lvec) 
    dummy_ind = Index(1; tags="dummyL_$(i)")
    return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
  end

  rvecs = map(enumerate(rvecs)) do (i,rvec) 
    dummy_ind = Index(1; tags="dummyR_$(i)")
    return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
  end

  length(rvecs) == 0 && return nothing,nothing,nothing,false

  ### make directsum out of the array of vectors
  U, newl = reduce((x,y) -> ITensors.directsum(
    x, y; tags=tags(commonind(centerwf[1],centerwf[2]))
  ), lvecs[2:end]; init=lvecs[1])
  V, newr = reduce((x,y) -> ITensors.directsum(
    x, y; tags=tags(commonind(centerwf[2],centerwf[3]))
  ), rvecs[2:end]; init=rvecs[1])


  # @show vals, newl, newr
  # @show vals, lvecs, rvecs
  # #TO DO construct U,S,V objects, using only the vals > cutoff, and at most maxdim

  # @show vals
  # @show uniqueinds(NL, L)
  # @show uniqueinds(NR, R)
  # @show inds(lvecs[1])
  # @show inds(rvecs[1])
  NL *= dag(U)
  NR *= dag(V)
  return NL, nothing, NR, true
end
