function general_expander(; expander_backend="two_site", svd_backend="svd", kwargs...)
  function expander(
    PH,
    psi::ITensorNetworks.AbstractTTN{Vert},
    b;
    maxdim,
    cutoff=1e-10,
    atol=1e-8,
    kws...,
  ) where {Vert}

    (typeof(b)!=NamedEdge{Int}) && return psi, PH

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
    elseif svd_backend == "krylov"
      svd_func = _krylov_svd_solve
    else
      error("svd_backend=$svd_backend not recognized (options are \"svd\" or \"krylov\")")
    end

    # atol refers to the tolerance in nullspace determination (for finite MPS can probably be set rather small)
    # cutoff refers to largest singular value of gradient (acceleration of population gain of expansion vectors) to keep
    # this should be related to the cutoff in two-site TDVP: \Delta_rho_i = 0.5 * lambda_y * tau **2 
    # note that in the initial SVD there is another cutoff parameter `cutoff_compress`, that we set to roughly machine precision for the time being
    # (allows to move bond-dimension between different partial QN sectors, if no such truncation occurs distribution of bond-dimensions
    # between different QNs locally is static once bond dimension saturates maxdim.)
    
    cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)

    # get subspace expansion
    psi = _expand_core(
      psi, PH, b, svd_func; maxdim, cutoff, cutoff_compress, atol, kwargs...,
    )

    PH  = position(PH, psi, [src(b),dst(b)])

    return psi, PH
  end
  return expander
end

function _svd_solve_normal(
  envMap, left_ind; maxdim, cutoff, kwargs...
)
  U, S, V = svd(
    ITensors.ITensorNetworkMaps.contract(envMap),
    left_ind;
    maxdim,
    cutoff,
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
    kwargs...,
  )

  vals = array(S)
  (length(vals) == 1 && vals[1]^2 ≤ cutoff) && return nothing,nothing,nothing

  return U,S,V
end

function _krylov_svd_solve(
  envMap, left_ind; maxdim, cutoff, kwargs...
)
  maxdim = min(maxdim, 15)
  envMapDag = adjoint(envMap)

  if !hasqns(left_ind)
    trial = randomITensor(eltype(envMap), left_ind)
    vals, lvecs, rvecs, info = KrylovKit.svdsolve(
      (x -> noprime(envMap * x), y -> noprime(envMapDag * y)), trial, # maxdim
    ) 

    ## cutoff unnecessary values and expand the resulting vectors by a dummy index
    vals = filter(v->v^2≥cutoff, vals)[1:min(maxdim,end)]
    (length(vals) == 0) && return nothing,nothing,nothing
    lvecs = lvecs[1:min(length(vals),end)]
    rvecs = rvecs[1:min(length(vals),end)]

    lvecs = map(enumerate(lvecs)) do (i,lvec) 
      dummy_ind = Index(1; tags="u") #tags(commonind(centerwf[1],centerwf[2])))
      return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
    end

    rvecs = map(enumerate(rvecs)) do (i,rvec) 
      dummy_ind = Index(1; tags="v") #tags(commonind(centerwf[1],centerwf[2])))
      return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
    end

    ### make directsum out of the array of vectors
    U,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="u" #tags(commonind(centerwf[1],centerwf[2]))
    ), lvecs[2:end]; init=lvecs[1])

    V,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="v" #tags(commonind(centerwf[2],centerwf[3]))
    ), rvecs[2:end]; init=rvecs[1])

    S = ITensors.diagITensor(vals, filter(v->hastags(v, "u"), inds(U)), filter(v->hastags(v, "v"), inds(V)))

  else
    vals_col  = Any[]
    lvecs_col = Any[]
    rvecs_col = Any[]
    d = Vector{real(eltype(envMap))}()

    for s in space(left_ind[1])
      last(s)==0 && continue

      theqn=first(s)
      trial = randomITensor(eltype(envMap), theqn, left_ind)
    
      vals, lvecs, rvecs, info = KrylovKit.svdsolve(
        (x -> noprime(envMap * x), y -> noprime(envMapDag * y)), trial, # maxdim
      )

      push!(vals_col,  vals)
      push!(lvecs_col, lvecs)
      push!(rvecs_col, rvecs)
      append!(d, vals)
    end

    d .= d .^ 2
    sort!(d; rev=true)
    truncerr, docut = truncate!(d; 
      cutoff, 
      use_relative_cutoff=false, 
      use_absolute_cutoff=true,
    )

    dropblocks = Int[]

    for (n,vals) in enumerate(vals_col)
      full_dim = length(vals)
      blockdim = 0
      val = vals[blockdim + 1]^2
      while blockdim + 1 ≤ full_dim && val > docut
        blockdim += 1
        (blockdim + 1 ≤ full_dim) && (val = vals[blockdim + 1]^2)
      end

      if blockdim == 0
        push!(dropblocks,n)
      else
        vals_col[n] .= vals[1:blockdim]
        lvecs_col[n] .= lvecs_col[n][1:blockdim]
        rvecs_col[n] .= rvecs_col[n][1:blockdim]
      end
    end

    deleteat!(vals_col,  dropblocks)
    deleteat!(lvecs_col, dropblocks)
    deleteat!(rvecs_col, dropblocks)

    (length(vals_col) == 0) && return nothing,nothing,nothing
    @show length(vals_col)

    map(lvecs_col) do lvecs
      @show storage(lvecs[1]).data
      @show inds(lvecs[1])
      # @show eachnzblock(storage(lvecs[1]).blockoffsets)
      # @show nzblocks(storage(lvecs[1]).blockoffsets)
      @show lvecs[1]
      dummy_ind = Index(1; tags="test") 
      @show NDTensors.BlockSparseTensor(ComplexF64, nzblocks(lvecs[1]),inds(lvecs[1]))
    end
    error("stop")

    lvecs_col = map(lvecs_col) do lvecs
      return map(lvecs) do lvec 
        dummy_ind = Index(1; tags="u") #tags(commonind(centerwf[1],centerwf[2])))
        return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
      end
    end

    # map(lvecs_col) do lvecs
    #   @show typeof(storage(lvecs[1][1]))
    #   @show lvecs
    # end

    rvecs = map(enumerate(rvecs)) do (i,rvec) 
      dummy_ind = Index(1; tags="v") #tags(commonind(centerwf[1],centerwf[2])))
      return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
    end

    ### make directsum out of the array of vectors
    U,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="u" #tags(commonind(centerwf[1],centerwf[2]))
    ), lvecs[2:end]; init=lvecs[1])

    V,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="v" #tags(commonind(centerwf[2],centerwf[3]))
    ), rvecs[2:end]; init=rvecs[1])

    S = ITensors.diagITensor(vals, filter(v->hastags(v, "u"), inds(U)), filter(v->hastags(v, "v"), inds(V)))

  end

  return U,S,V
end

function _two_site_expand_core(
  psi, PH, b, svd_func; maxdim, cutoff, cutoff_compress, atol, kwargs...,
)
  n1, n2 = src(b), dst(b)
  g = underlying_graph(PH)

  U,S,V = svd(
    psi[n1], uniqueinds(psi[n1], psi[n2]); maxdim=maxdim, cutoff=cutoff_compress,
  )

  phi_1 = U
  phi_2 = psi[n2] * V
  bondtensor = S
  old_linkdim = dim(commonind(U, S))

  # don't expand if we are already at maxdim
  (old_linkdim >= maxdim) && return psi

  linkind_l = commonind(phi_1, bondtensor)
  linkind_r = commonind(phi_2, bondtensor)

  # compute nullspace to the left and right 
  NL = nullspace(phi_1, linkind_l; atol=atol)
  NR = nullspace(phi_2, linkind_r; atol=atol)

  # if nullspace is empty (happen's for product states with QNs)
  (norm(NL) == 0.0 || norm(NR) == 0.0) && return psi

  PH = set_nsite(PH, 2)
  PH = position(PH, psi, [n1,n2])

  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn1 = noprime(reduce(*, PHn1, init=phi_1*PH.H[n1]))
  PHn2 = map(e -> PH.environments[NamedEdge(e)], [v => n2 for v in neighbors(g, n2) if v != n1])
  PHn2 = noprime(reduce(*, PHn2, init=bondtensor*phi_2*PH.H[n2]))

  ininds = uniqueinds(NR,phi_2)
  outinds = uniqueinds(NL,phi_1)
  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([NL,PHn1,PHn2,NR], ininds, outinds)

  norm(ITensors.ITensorNetworkMaps.contract(envMap)) ≤ 1e-13 && return psi

  U,S,V= svd_func(envMap, outinds; maxdim=maxdim-old_linkdim, cutoff=cutoff)
  isnothing(U) && return psi

  @assert dim(commonind(U, S)) ≤ maxdim
  @show NL

  NL *= dag(U)
  NR *= dag(V)

  # expand current site tensors
  new_psi_1, newl = ITensors.directsum(
    phi_1 => uniqueinds(phi_1, NL), dag(NL) => uniqueinds(NL, phi_1); tags=(tags(commonind(psi[n1],psi[n2])),)
  )
  new_psi_2, newr = ITensors.directsum(
    phi_2 => uniqueinds(phi_2, NR), dag(NR) => uniqueinds(NR, phi_2); tags=(tags(commonind(psi[n1],psi[n2])),)
  )

  @assert dim(newl) <= maxdim
  @assert dim(newr) <= maxdim
  
  # zero-pad bond-tensor (the orthogonality center)
  new_bondtensor = ITensor(dag(newl)..., dag(newr)...)
  map(eachindex(bondtensor)) do I
    v = bondtensor[I]
    !iszero(v) && (return new_bondtensor[I]=v)
  end

  psi[n2] = new_psi_2
  psi[n1] = noprime(new_psi_1 * new_bondtensor)

  return psi
end

function _full_expand_core(
  # psi, PH, b, svd_func; maxdim, cutoff, cutoff_compress, atol, kwargs...,
  psi::ITensorNetworks.AbstractTTN{Vert},
  PH,
  b,
  svd_func;
  expander_cache=Any[],
  maxdim,
  cutoff,
  cutoff_compress,
  atol=1e-8,
  kwargs...,
 ) where {Vert}

  if isempty(expander_cache)
    @warn("building environment of H^2 from scratch!")
    # build H^2 out of H
    new_vertex_data = replaceprime.(map(*, vertex_data(data_graph(PH.H)), prime.(vertex_data(data_graph(PH.H)))), 2 => 1)
    H2 = TTN(ITensorNetwork(DataGraph(underlying_graph(PH.H), new_vertex_data, edge_data(data_graph(PH.H)))), PH.H.ortho_center)
    push!(expander_cache, ProjTTN(H2))
  end

  PH2 = expander_cache[1]

  n1, n2 = src(b), dst(b)
  cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)
  g = underlying_graph(PH)

  U, S, V = svd(
    psi[n1], uniqueinds(psi[n1], psi[n2]); maxdim=maxdim, cutoff=cutoff_compress, kwargs...
  )

  phi_1 = U
  phi_2 = psi[n2] * V
  bondtensor = S
  old_linkdim = dim(commonind(U, S))

  # don't expand if we are already at maxdim
  old_linkdim >= maxdim && return psi

  # compute nullspace to the left and right 
  linkind_l = commonind(phi_1, S)
  Nn1 = nullspace(phi_1, linkind_l; atol=atol)

  # if nullspace is empty (happen's for product states with QNs)
  norm(Nn1) == 0.0 && return psi

  PH  = position(PH, psi, [n1])
  PH2 = position(PH2, psi, [n1])

  ## compute both environments
  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn1 = noprime(reduce(*, PHn1, init=bondtensor*phi_1*PH.H[n1]))*Nn1
  PHn2 = map(e -> PH2.environments[NamedEdge(e)], [v => n2 for v in neighbors(g, n2) if v != n1])
  PHn2 = reduce(*, PHn2, init=(phi_2*PH2.H[n2]*adjoint(phi_2)))

  outinds = commoninds(Nn1, PHn1)
  ininds = adjoint.(outinds)
  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([PHn1,PHn2,adjoint(PHn1)], ininds, outinds)

  norm(ITensors.ITensorNetworkMaps.contract(envMap)) ≤ 1e-13 && return psi

  # svd-decomposition
  U,S,V= svd_func(envMap, outinds; maxdim=maxdim-old_linkdim, cutoff=cutoff)
  isnothing(U) && return psi

  @assert dim(commonind(U, S)) ≤ maxdim

  newL = Nn1*dag(U)

  # expand current site tensors
  new_psi_1, newl = ITensors.directsum(
    phi_1 => uniqueinds(phi_1, newL), dag(newL) => uniqueinds(newL, phi_1); tags=(tags(commonind(psi[n1],psi[n2])),)
  )

  @assert dim(newl) <= maxdim

  # zero-pad bond-tensor (the orthogonality center)
  new_bondtensor = ITensor(dag(newl)..., commonind(bondtensor, phi_2))
  map(eachindex(bondtensor)) do I
    v = bondtensor[I]
    !iszero(v) && (return new_bondtensor[I]=v)
  end

  psi[n1] = noprime(new_psi_1)
  psi[n2] = phi_2*new_bondtensor
  PH2 = position(PH2, psi, [n1,n2])

  return psi
end
