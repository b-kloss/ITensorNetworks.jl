function general_expander(; expander_backend="two_site", svd_backend="svd", kwargs...)
  function expander(
    PH,
    ψ::ITensorNetworks.AbstractTTN{Vert},
    b;
    # maxdim,
    # cutoff,
    # atol=1e-8,
    # kws...,
  ) where {Vert}
    maxdim=20
    cutoff=1E-10
    atol=1e-8
    @show b
    (typeof(b)!=NamedEdge{Int}) && return ψ, PH

    # determine which expansion method and svd method to use
    if expander_backend == "two_site"
      _expand_core = _two_site_expand_core
    elseif expander_backend == "full"
      _expand_core = _full_expand_core
    else
      error("expander_backend=$expander_backend not recognized (options are \"2-site\" or \"full\")")
    end

    if svd_backend == "svd"
      svd_func = _svd_solve_normal
    elseif svd_backend == "full"
      svd_func = _krylov_svd_solve
    else
      error("svd_backend=$svd_backend not recognized (options are \"svd\" or \"Krylov svd_solve\")")
    end

    # atol refers to the tolerance in nullspace determination (for finite MPS can probably be set rather small)
    # cutoff refers to largest singular value of gradient (acceleration of population gain of expansion vectors) to keep
    # this should be related to the cutoff in two-site TDVP: \Delta_rho_i = 0.5 * lambda_y * tau **2 
    # note that in the initial SVD there is another cutoff parameter `cutoff_compress`, that we set to roughly machine precision for the time being
    # (allows to move bond-dimension between different partial QN sectors, if no such truncation occurs distribution of bond-dimensions
    # between different QNs locally is static once bond dimension saturates maxdim.)
    #
    cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)
    
    n1 = src(b)
    n2 = dst(b)

    U, S, V = svd(
      ψ[n1], uniqueinds(ψ[n1], ψ[n2]); maxdim=maxdim, cutoff=cutoff_compress, kwargs...
    )

    ϕ_1 = U
    ϕ_2 = ψ[n2] * V
    bondtensor = S
    old_linkdim = dim(commonind(U, S))

    # don't expand if we are already at maxdim
    (old_linkdim >= maxdim) && (return nothing)

    # orthogonalize(ψ,n1)
    linkind_l = commonind(ϕ_1, bondtensor)
    linkind_r = commonind(ϕ_2, bondtensor)

    # compute nullspace to the left and right 
    NL = nullspace(ϕ_1, linkind_l; atol=atol)
    NR = nullspace(ϕ_2, linkind_r; atol=atol)

    # if nullspace is empty (happen's for product states with QNs)
    (norm(NL) == 0.0 || norm(NR) == 0.0) && return ψ, PH

    # form 2-site wavefunction
    ϕ = [ϕ_1, bondtensor, ϕ_2]
    PH = position(PH, ψ, b)

    # get subspace expansion
    newL, S, newR, success = _expand_core(
      ϕ, PH, NL, NR, b, svd_func; maxdim=maxdim-old_linkdim, cutoff, kwargs...
    )

    # think about adaptive expansion schemes etc
    success || return ψ, PH

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

    @show PH
    PH = position(PH, ψ, b)
    println("expanded")
    return ψ, PH
  end
  return expander
end

function _svd_solve_normal(
  ϕs::Vector{ITensor}, PH, NL, NR; maxdim, cutoff, kwargs...
)
  ϕ = reduce(*, ϕs)
  _svd_solve_normal(ϕ, PH, NL, NR; maxdim, cutoff, kwargs...)
end

function _svd_solve_normal(
  ϕ::ITensor, PH, NL, NR; maxdim, cutoff, kwargs...
  )
  if length(PH)==1 
    ϕH = noprime(apply(PH[1],ϕ))   
  else 
    @assert length(PH)>1
    ϕH = reduce(*, PH; init=ϕ)
  end

  ϕH = NL * ϕH * NR
  (norm(ϕH) == 0.0) && @warn("method not working")

  U, S, V = svd(
    ϕH,
    commoninds(ϕH, NL);
    maxdim,
    cutoff,
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
    kwargs...,
  )

  return U,S,V
end

function _krylov_svd_solve(
  ϕs::Vector{ITensor}, PH, NL, NR; maxdim, cutoff, kwargs...
  )
  ϕ = reduce(*, ϕs)
  _krylov_svd_solve(ϕ, PH, NL, NR; maxdim, cutoff, kwargs...)
end

function _krylov_svd_solve(
  ϕ::ITensor, PH, NL, NR; maxdim, cutoff, kwargs...
  )
  ### why that??? ###
  outinds = uniqueinds(NL, NL)
  ininds = uniqueinds(NR, NR)

  if length(PHs)==1 
    B2 = ITensorNetworkMaps.ITensorNetworkMap([NR, noprime(env(ϕ)), NL], ininds, outinds)
  else 
    @assert length(PH)>1
    B2 = ITensorNetworkMaps.ITensorNetworkMap([NR, ϕ, PH..., NL], ininds, outinds)
  end

  # env_n1 = reduce(*, PHn1, init=centerwf[1])
  # env_n1 *= PHsiteMPOs[1]
  # env_n2 = reduce(*, PHn2, init=centerwf[3])
  # env_n2 *= PHsiteMPOs[2]
  # env_n2 *= centerwf[2]  ##contract C into one of them because otherwise the application is more costly?
  # env_n2 = noprime(env_n2)
  # env_n1 = noprime(env_n1)

  B2dag = adjoint(B2)

  ### think about that
  trial = randomITensor(eltype(ϕ), filter(i->hastags(i, "n"), inds(NL)))

  vals, lvecs, rvecs, info = KrylovKit.svdsolve(
    (x -> noprime(B2 * x), y -> noprime(B2dag * y)), trial, tol=cutoff,  
  )    

  ### cutoff unnecessary values and expand the resulting vectors by a dummy index
  vals = filter(v->v≥cutoff, vals)[1:min(maxdim,end)]
  (length(vals) == 0) && return nothing,nothing,nothing,false
  lvecs = lvecs[1:min(length(vals),end)]
  rvecs = rvecs[1:min(length(vals),end)]

  lvecs = map(enumerate(lvecs)) do (i,lvec) 
    dummy_ind = Index(1; tags="dummyL_$(i)")
    return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
  end

  rvecs = map(enumerate(rvecs)) do (i,rvec) 
    dummy_ind = Index(1; tags="dummyR_$(i)")
    return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
  end

  ### make directsum out of the array of vectors
  U, newl = reduce((x,y) -> ITensors.directsum(
    x, y; tags=tags(commonind(centerwf[1],centerwf[2]))
  ), lvecs[2:end]; init=lvecs[1])
  V, newr = reduce((x,y) -> ITensors.directsum(
    x, y; tags=tags(commonind(centerwf[2],centerwf[3]))
  ), rvecs[2:end]; init=rvecs[1])
  S = ITensors.diagITensor(vals, commonind(centerwf[1],centerwf[2]), commonind(centerwf[2],centerwf[3]))

  return U,S,V
end

function _two_site_expand_core(
  centerwf::Vector{ITensor}, PH, NL, NR, b, svd_func; maxdim, cutoff, kwargs...
)
  U,S,V = _svd_solve_normal(centerwf, [PH], NL, NR; maxdim=maxdim, cutoff=cutoff, kwargs...)

  @assert dim(commonind(U, S)) <= maxdim

  NL *= dag(U)
  NR *= dag(V)
  return NL, S, NR, true
end

function _two_site_krylov_expand_core(
  centerwf::Vector{ITensor}, PH, NL, NR, b; maxdim, cutoff, kwargs...
)
  (n1,n2) = b
  PH = position(PH, ψ, b)
  g = underlying_graph(PH)
  maxdim = min(maxdim, 15)

  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn2 = map(e -> PH.environments[NamedEdge(e)], [v => n2 for v in neighbors(g, n2) if v != n1])
  PHsiteMPOs = map(n -> PH.H[n], b)

  U,S,V = _krylov_svd_solve(centerwf, [PHn1,PHn2,PHsiteMPOs], NL, NR; maxdim=maxdim, cutoff=cutoff, kwargs...)

  NL *= dag(U)
  NR *= dag(V)
  return NL, S, NR, true
end

function _full_expand_core!(
  ψ::ITensorNetworks.AbstractTTN{Vert},
  PH,
  PH2,
  b::Vector{Vert};
  bondtensor=nothing,
  maxdim,
  cutoff,
  atol=1e-8,
  kwargs...,
 ) where {Vert}

  PH  = position(PH, ψ, [n1])
  PH2 = position(PH2, ψ, [n1])

  # if nullspace is empty (happen's for product states with QNs)
  norm(Nn1) == 0.0 && return bondtensor

  ## brute-force for now, need a better way to do that
  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn1 = noprime(reduce(*, PHn1, init=ϕ_1*PH.H[n1])) * Nn1
  PHn2 = map(e -> PH2.environments[NamedEdge(e)], [v => n2 for v in neighbors(g, n2) if v != n1])
  PHn2 = reduce(*, PHn2, init=(ϕ_2*PH2.H[n2]*prime(dag(ϕ_2)))*bondtensor*prime(dag(bondtensor)))

  # if n2 > n1
  #   PHn2 = mapreduce(*, filter(x->x>n2, vertices(ψ)); init=ϕ_2*PH.H[n2]) do v
  #     return ψ[v]*PH.H[v]
  #   end
  # else
  #   PHn2 = mapreduce(*, filter(x->x<n2, vertices(ψ)); init=ϕ_2*PH.H[n2]) do v
  #     return ψ[v]*PH.H[v]
  #   end
  # end

  ϕH = PHn1*PHn2*prime(dag(PHn1))  

  norm(ϕH) == 0. && return nothing

  # get subspace expansion
  # newL, S, newR, success = _subspace_expand_full_core(
  #   ϕ, PHn1, PHn2, Nn1; maxdim=maxdim - old_linkdim, cutoff, kwargs...
  # )
  
  U, S, V = svd(
    ϕH,
    commoninds(ϕH, Nn1);
    maxdim=maxdim - old_linkdim,
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
