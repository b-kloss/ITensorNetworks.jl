function general_expander(; expander_backend="two_site", svd_backend="svd", kwargs...)
  function expander(
    PH,
    psi::ITensorNetworks.AbstractTTN{Vert},
    phi,
    sweep_step,
    direction;
    maxdim,
    cutoff=1e-10,
    atol=1e-8,
    expand_dir=-1, # +1 for future direction, -1 for past
    kws...,
  ) where {Vert}

    ### only on edges
    # (typeof(ITensorNetworks.pos(sweep_step))!=NamedEdge{Int}) && return psi, phi, PH
    ### only on verts
    (typeof(ITensorNetworks.pos(sweep_step))==NamedEdge{Int}) && return psi, phi, PH

    # determine which expansion method and svd method to use
    if expander_backend == "none"
      return psi, phi, PH
    elseif expander_backend == "full"
      _expand_core = _full_expand_core
    elseif expander_backend == "two_site"
      _expand_core = _two_site_expand_core
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

    if typeof(ITensorNetworks.pos(sweep_step)) == NamedEdge{Int} 
      verts = [src(ITensorNetworks.pos(sweep_step)),dst(ITensorNetworks.pos(sweep_step))]
    else

      ## kind of hacky - only works for mps. More general?
      n1 = first(ITensorNetworks.pos(sweep_step))
      if isforward(direction) 
        n2 = n1 + 1
        n2 > 100 && return psi,phi,PH
      else
        n2 = n1 - 1
        n2 < 1 && return psi,phi,PH
      end
      verts = [n1,n2]

      ## bring it into the same functional form used for the Named-Edge case
      left_inds = uniqueinds(psi[n1], psi[n2])
      U, S, V = svd(psi[n1], left_inds; lefttags=tags(commonind(psi[n1],psi[n2])), righttags=tags(commonind(psi[n1],psi[n2])))
      psi[n1] = U
      phi = S*V
    end

    # get subspace expansion
    psi, phi, PH = _expand_core(
      PH, psi, phi, verts, svd_func; expand_dir, maxdim, cutoff, cutoff_compress, atol, kwargs...,
    )

    if typeof(ITensorNetworks.pos(sweep_step)) != NamedEdge{Int} 
      phi = psi[first(ITensorNetworks.pos(sweep_step))] * phi
    end

    PH = set_nsite(PH, nsite(sweep_step))
    PH = position(PH, psi, pos(sweep_step))

    return psi, phi, PH
  end
  return expander
end

function _svd_solve_normal(
  envMap, left_ind; maxdim, cutoff, envflux, kwargs...
)
  U,S,V = svd(
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

function _build_USV_without_QN(vals, lvecs, rvecs)

  # attach trivial index to left/right eigenvectors to take directsum over it
  lvecs = map(lvecs) do lvec 
    dummy_ind = Index(1; tags="u") 
    return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
  end

  rvecs = map(rvecs) do rvec 
    dummy_ind = Index(1; tags="v")
    return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
  end

  ### perform directsum
  U,_ = reduce((x,y) -> ITensors.directsum(
    x, y; tags="u", 
  ), lvecs[2:end]; init=lvecs[1])

  V,_ = reduce((x,y) -> ITensors.directsum(
    x, y; tags="v",
  ), rvecs[2:end]; init=rvecs[1])

  S = ITensors.diagITensor(vals, filter(v->hastags(v, "u"), inds(U)), filter(v->hastags(v, "v"), inds(V)))

  return U,S,V
end

function _build_USV_with_QN(vals_col, lvecs_col, rvecs_col, remainingSpaces;envflux)
  # attach trivial index to left/right eigenvectors to take directsum over it
  lvecs_col = map(zip(remainingSpaces,lvecs_col)) do (s,lvecs)
    return map(lvecs) do lvec 
      dummy_ind = Index(Pair(first(s),1); tags="u", dir=ITensors.In)
      return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
    end
  end

  rvecs_col = map(zip(remainingSpaces,rvecs_col)) do (s,rvecs)
    return map(rvecs) do rvec
      rvec=rvec 
      dummy_ind = Index(Pair(envflux-first(s),1); tags="v", dir=ITensors.In)
      return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
    end
  end

  ### perform directsum of left/right eigenvectors for each sector
  lvecs_col = map(lvecs_col) do lvecs
    l, inds_l = reduce((x,y) -> ITensors.directsum(
      x, y; tags="u",
    ), lvecs[2:end]; init=lvecs[1])
    length(lvecs) == 1 && return l => inds_l
    C = combiner(inds_l; tags="u", dir=dir(inds_l))
    l = l*C
    return l => commonind(l,C)
  end

  rvecs_col = map(rvecs_col) do rvecs
    r, inds_r = reduce((x,y) -> ITensors.directsum(
      x, y; tags="v",
    ), rvecs[2:end]; init=rvecs[1])
    length(rvecs) == 1 && return r => inds_r
    C = combiner(inds_r; tags="v", dir=dir(inds_r))
    r = r*C
    return r => commonind(r,C)
  end

  ### perform directsum over all sectors
  U,_ = reduce((x,y) -> ITensors.directsum(
    x, y; tags="u" 
  ), lvecs_col[2:end]; init=lvecs_col[1])

  V,_ = reduce((x,y) -> ITensors.directsum(
    x, y; tags="v",
  ), rvecs_col[2:end]; init=rvecs_col[1])

  S = ITensors.diagITensor(vcat(vals_col...), filter(v->hastags(v, "u"), dag(inds(U))), filter(v->hastags(v, "v"), dag(inds(V))))
  return U,S,V
end

function _truncate_blocks!(d, vals_col, lvecs_col, rvecs_col, remainingSpaces, cutoff, maxdim) 
  d .= d .^ 2
  sort!(d; rev=true)
  truncerr, docut = truncate!(d; 
    cutoff, 
    maxdim,
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
      vals_col[n] = vals[1:blockdim]
      lvecs_col[n] = lvecs_col[n][1:blockdim]
      rvecs_col[n] = rvecs_col[n][1:blockdim]
    end
  end

  deleteat!(vals_col,  dropblocks)
  deleteat!(lvecs_col, dropblocks)
  deleteat!(rvecs_col, dropblocks)
  deleteat!(remainingSpaces,  dropblocks)
end

function _krylov_svd_solve(
  envMap, left_ind; maxdim, cutoff, envflux, kwargs...
)
  maxdim = min(maxdim, 15)
  envMapDag = adjoint(envMap)

  if !hasqns(left_ind)
    trial = randomITensor(eltype(envMap), left_ind)

    if ! _kkit_init_check(trial, envMapDag,envMap)
      return nothing,nothing,nothing
    end

    try
      vals, lvecs, rvecs, info = KrylovKit.svdsolve(
        (x -> envMap * x, y -> envMapDag * y), trial,
      )
    catch e 
      @show e
      return _svd_solve_normal(envMap, left_ind; maxdim, cutoff)
    end

    ## cutoff unnecessary values and expand the resulting vectors by a dummy index
    vals = filter(v->v^2≥cutoff, vals)[1:min(maxdim,end)]
    (length(vals) == 0) && return nothing,nothing,nothing
    lvecs = lvecs[1:min(length(vals),end)]
    rvecs = rvecs[1:min(length(vals),end)]

    U,S,V = _build_USV_without_QN(vals, lvecs, rvecs)

  else
    vals_col  = Any[]
    lvecs_col = Any[]
    rvecs_col = Any[]
    d = Vector{real(eltype(envMap))}()
    remainingSpaces = Any[]

    for s in space(left_ind[1])
      last(s)==0 && continue

      theqn=first(s)
      trial = randomITensor(eltype(envMap), theqn, left_ind)

      ##some checks to not pass singular/zero-size problem to KrylovKit
      adjtrial=envMapDag(trial)
      trial2=envMap(adjtrial)
      if size(storage(adjtrial))==(0,) || size(storage(trial2))==(0,)
        continue
      elseif ! _kkit_init_check(trial, envMapDag,envMap)
        continue
      end

      try
        vals, lvecs, rvecs, info = KrylovKit.svdsolve(
          (x -> (noprime((envMap) * x)), y -> (envMapDag * y)), trial,
        )
      catch e 
        @show e
        return _svd_solve_normal(envMap, left_ind; maxdim, cutoff,envflux)
      end
      push!(vals_col,  vals)
      push!(lvecs_col, lvecs)
      push!(rvecs_col, conj.(dag.(rvecs)))  ###is the conjugate here justified or not?
      push!(remainingSpaces, s)
      append!(d, vals)
    end
    (length(d) == 0) && return nothing,nothing,nothing

    _truncate_blocks!(d, vals_col, lvecs_col, rvecs_col, remainingSpaces, cutoff, maxdim)

    (length(vals_col) == 0) && return nothing,nothing,nothing

    U,S,V = _build_USV_with_QN(vals_col, lvecs_col, rvecs_col, remainingSpaces;envflux=envflux)
  end

  return U,S,V
end

function _two_site_expand_core(
  PH, psi, phi, verts, svd_func; expand_dir, maxdim, cutoff, cutoff_compress, atol, kwargs...,
)
  maxdim = 20

  psis = map(n -> psi[n], verts)
  old_linkdim = dim(commonind(first(psis), phi))

  PH = set_nsite(PH, 2)
  PH = position(PH, psi, verts)

  # don't expand if we are already at maxdim
  (old_linkdim >= maxdim) && return psi, phi, PH

  linkinds = map(psi -> commonind(psi, phi), psis)

  # compute nullspace to the left and right 
  nullVecs = map(zip(psis,linkinds)) do (psi,linkind)
    return nullspace(psi, linkind; atol=atol)
  end
  
  # if nullspace is empty (happen's for product states with QNs)
  sum(norm.(nullVecs) .== 0) > 0 && return psi, phi, PH

  ## build environments
  g = underlying_graph(PH)
  envs = map(zip(verts,psis)) do (n,psi)
    return noprime(mapreduce(*, [v => n for v in neighbors(g, n) if !(v ∈ verts)]; init = psi*PH.H[n]) do e
      return PH.environments[NamedEdge(e)]
    end )
  end

  ininds = uniqueinds(last(nullVecs),last(psis))
  outinds = uniqueinds(first(nullVecs),first(psis))

  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([last(nullVecs),last(envs),phi,first(envs),first(nullVecs)], outinds, ininds)
  envMapc = ITensors.ITensorNetworkMaps.contract(envMap)
  
  norm(envMapc) ≤ 1e-13 && return psi, phi, PH
  U,S,V = svd_func(envMap, outinds; maxdim=maxdim-old_linkdim, cutoff=cutoff,envflux=flux(envMapc))
  isnothing(U) && return psi, phi, PH

  @assert dim(commonind(U, S)) ≤ maxdim-old_linkdim

  nullVecs[1] = nullVecs[1] * dag(U)
  nullVecs[2] = nullVecs[2] * dag(V)

  # expand current site tensors
  new_psis = map(zip(psis, nullVecs)) do (psi,nullVec)
    return ITensors.directsum(
      psi => uniqueinds(psi, nullVec), dag(nullVec) => dag(uniqueinds(nullVec, psi)); tags=(tags(commonind(psi,phi)),)
    )
  end

  new_inds = [last(x)[1]  for x in new_psis]
  new_psis = [first(x) for x in new_psis]

  combiners = map(new_ind -> combiner(new_ind; tags=tags(new_ind), dir=dir(new_ind)), new_inds)

  @assert sum(findmax.(dim.(new_inds))[1] .> maxdim) == 0

  # zero-pad bond-tensor (the orthogonality center)
  if hasqns(phi)
    new_phi=ITensor(eltype(phi),flux(phi),dag(last(new_inds)),dag(first(new_inds)))
    fill!(new_phi,0.0)
  else
    new_phi = ITensor(eltype(phi),dag(first(new_inds)),dag(last(new_inds)))
  end

  map(eachindex(phi)) do I
    v = phi[I]
    !iszero(v) && (return new_phi[I]=v)
  end

  old_twosite_tensor = psi[last(verts)]*phi*psi[first(verts)]

  for (v,new_psi,C) in zip(verts,new_psis,combiners)
    psi[v] = noprime(new_psi*C)
  end

  new_phi = dag(first(combiners)) * new_phi * dag(last(combiners))

  @assert norm(psi[last(verts)]*new_phi*psi[first(verts)] - old_twosite_tensor) <= 1e-10

  # if expand_dir == +1
  #   U,S,V = svd(psi[n1]*new_phi, uniqueinds(psi[n1], new_phi); lefttags = tags(commonind(psi[n1],new_phi)), righttags = tags(commonind(psi[n2],new_phi))) 
  #
  #   psi[n1] = U
  #   new_phi = S*V
  # end
 
  return psi, new_phi, PH
end

function _full_expand_core(
  PH, psi, phi, pos, svd_func; expand_dir, maxdim, cutoff, cutoff_compress, atol, expander_cache=Any[],kwargs..., 
) where {Vert}

  if isempty(expander_cache)
    @warn("building environment of H^2 from scratch!")

    g = underlying_graph(PH.H)
    H = vertex_data(data_graph(PH.H))

    H_dag = swapprime.(prime.(dag.(H)), 1,2, tags = "Site")
    H2_vd= replaceprime.(map(*, H, H_dag), 2 => 1)
    H2_ed = edge_data(data_graph(PH.H))

    H2 = TTN(ITensorNetwork(DataGraph(g, H2_vd, H2_ed)), PH.H.ortho_center)

    push!(expander_cache, ProjTTN(H2))
  end

  PH2 = expander_cache[1]
  n1 = expand_dir>0 ? pos[2] : pos[1]
  n2 = expand_dir>0 ? pos[1] : pos[2]

  PH2 = set_nsite(PH2, 1)
  PH2 = position(PH2, psi, [n1])

  psi1 = psi[n1]
  psi2 = psi[n2]
  old_linkdim = dim(commonind(psi1, phi))

  # don't expand if we are already at maxdim
  old_linkdim >= maxdim && return psi, phi, PH

  # compute nullspace to the left and right 
  linkind_l = commonind(psi1, phi)
  nullVec = nullspace(psi1, linkind_l; atol=atol)

  # if nullspace is empty (happen's for product states with QNs)
  norm(nullVec) == 0.0 && return psi, phi, PH

  PH = position(PH, psi, [n1])

  ## compute both environments
  g = underlying_graph(PH)
  PHn1 = map(e -> PH.environments[NamedEdge(e)], [v => n1 for v in neighbors(g, n1) if v != n2])
  PHn1 = noprime(reduce(*, PHn1, init=phi*psi1*PH.H[n1]))*nullVec
  PHn2 = map(e -> PH2.environments[NamedEdge(e)], [v => n2 for v in neighbors(g, n2) if v != n1])
  PHn2 = reduce(*, PHn2, init=(psi2*PH2.H[n2]*prime(dag(psi2))))

  outinds = uniqueinds(nullVec,psi1)
  ininds = dag.(outinds)

  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([PHn1,PHn2,prime(dag(PHn1))] , outinds, ininds)
  envMapc=ITensors.ITensorNetworkMaps.contract(envMap)
  norm(envMapc) ≤ 1e-13 && return psi,phi,PH

  # svd-decomposition
  U,S,_= svd_func(envMap, outinds; maxdim=maxdim-old_linkdim, cutoff=cutoff,envflux=flux(envMapc))
  isnothing(U) && return psi,phi,PH

  @assert dim(commonind(U, S)) ≤ maxdim

  newL = nullVec*dag(U)

  # expand current site tensors
  new_psi, newl = ITensors.directsum(
    psi1 => uniqueinds(psi1, newL), dag(newL) => uniqueinds(newL, psi1); tags=(tags(commonind(psi1,phi)),)
  )
  Cl = combiner(newl; tags=tags(newl[1]), dir=dir(newl[1]))

  @assert dim(newl) <= maxdim

  # zero-pad bond-tensor (the orthogonality center)
  if hasqns(phi)
    new_phi=ITensor(eltype(phi),flux(phi),commonind(phi,psi2),dag(newl)...)
    fill!(new_phi,0.0)
  else
    new_phi = ITensor(eltype(phi),dag(newl)...,commonind(phi,psi2))
  end

  map(eachindex(phi)) do I
    v = phi[I]
    !iszero(v) && (return new_phi[I]=v)
  end
  old_twosite_tensor=psi[n2]*phi*psi[n1]
  psi[n1] = noprime(new_psi*Cl)
  @assert norm(psi[n2]*(dag(Cl)*new_phi)*psi[n1] - old_twosite_tensor)<=1e-13

  if expand_dir == +1
    U,S,V = svd(psi[n1]*new_phi, uniqueinds(psi[n1], new_phi); lefttags = tags(commonind(psi[n1],new_phi)), righttags = tags(commonind(psi[n2],new_phi))) 

    psi[n1] = U
    new_phi = S*V
  end
  
  return psi, dag(Cl)*new_phi, PH
end

function _kkit_init_check(u₀,theadj,thenormal)
  β₀ = norm(u₀)
  iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
  v₀ = theadj(u₀)
  α = norm(v₀) / β₀
  Av₀ = thenormal(v₀) # apply operator
  α² = dot(u₀, Av₀) / β₀^2
  if norm(α²) < eps(Float64)
    return false
  else
    α² ≈ α * α || throw(ArgumentError("operator and its adjoint are not compatible"))
  end
  return true
end
