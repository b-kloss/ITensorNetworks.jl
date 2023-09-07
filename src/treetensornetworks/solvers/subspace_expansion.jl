# function general_expander(; expander_backend="none", svd_backend="svd", kwargs...)
#   function expander(
#     PH,
#     psi::ITensorNetworks.AbstractTTN{Vert},
#     phi,
#     region,
#     direction;
#     maxdim,
#     cutoff=1e-10,
#     atol=1e-8,
#     kwargs...,
#   ) where {Vert}
#
#     cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)
#     expander_cache = get(kwargs, :expander_cache, Any[])
#     nsites = (region isa AbstractEdge) ? 0 : length(region)
#
#     # determine which expansion method and svd method to use
#     if expander_backend == "none"
#       return psi, phi, PH
#     elseif expander_backend == "full"
#       _expand_core = _full_expand_core_vertex
#       expand_dir = get(kwargs, :expand_dir, -1)
#       
#     elseif expander_backend == "two_site"
#       _expand_core = _two_site_expand_core
#       expand_dir = get(kwargs, :expand_dir, +1)
#       #enforce expand_dir in the tested direction
#       @assert expand_dir==1
#       
#     else
#       error("expander_backend=$expander_backend not recognized (options are \"2-site\" or \"full\")")
#     end
#
#     if svd_backend == "svd"
#       svd_func = _svd_solve_normal
#     elseif svd_backend == "krylov"
#       svd_func = _krylov_svd_solve
#     else
#       error("svd_backend=$svd_backend not recognized (options are \"svd\" or \"krylov\")")
#     end
#
#     cutoff_compress = get(kwargs, :cutoff_compress, 1e-12)
#     expander_cache = get(kwargs, :expander_cache, Any[])
#     
#     to = get(kwargs, :to, Any[])
#
#     # subspace expansion
#     psi, phi, PH = _expand_core(
#       PH, psi, phi, region, svd_func; expand_dir, expander_cache, maxdim, cutoff, cutoff_compress, atol, to,
#     )
#
#
#     # update environment
#     PH = set_nsite(PH, nsites)
#     PH = position(PH, psi, region)
#
#     return psi, phi, PH
#   end
#   return expander
# end



#_svd_solve_normal(envMap,envMapDag, left_ind;kwargs...)=svd_solve_normal(envMap, left_ind;maxdim=get(::maxdim,kwargs),cutoff=cutoff)
#_svd_solve_normal(T,envMap,envMapDag, left_ind;kwargs...)=svd_solve_normal(envMap, left_ind;maxdim=maxdim,cutoff=cutoff)




function implicit_nullspace(A, linkind)
  ###only works when applied in the direction of the environment tensor, not the other way (due to use of ITensorNetworkMap which is not reversible)
  outind=uniqueinds(A,linkind)
  inind=outind  #?
  P=ITensors.ITensorNetworkMaps.ITensorNetworkMap([prime(dag(A),linkind),prime(A,linkind)],inind,outind)
  return x::ITensor -> x-P(x)
end

function _two_site_expand_core(
  PH, psi, phi0, region, svd_func; direction, expand_dir=+1, maxdim, cutoff, atol=1e-8, kwargs...,
)
  #enforce expand_dir in the tested direction
  # @assert expand_dir==+1

  (typeof(region)==NamedEdge{Int}) && return psi, phi0, PH
  if typeof(region) == NamedEdge{Int} 
    n1,n2 = (src(region),dst(region))
    verts = expand_dir == 1 ? [n1,n2] : [n2,n1]
    psis = map(n -> psi[n], verts)
    phi=copy(phi0)
  else

    ## kind of hacky - only works for mps. More general?
    n1 = first(region)
    theflux=flux(psi[n1])
    if direction == 1 
      n2 = expand_dir == 1 ? n1-1 : n1+1
    else
      n2 = expand_dir == 1 ? n1+1 : n1-1
    end
    (n2 < 1 || n2 > length(vertices(psi))) && return psi,phi0,PH
    verts = [n1,n2]
    psis = map(n -> psi[n], verts)
    ## bring it into the same functional form used for the Named-Edge case by doing an svd
    left_inds = uniqueinds(psi[n1], psi[n2])
    U, S, V = svd(psis[findall(verts.==n1)[]], left_inds; lefttags=tags(commonind(psi[n1],psi[n2])), righttags=tags(commonind(psi[n1],psi[n2])))
    psis[findall(verts.==n1)[]]= U
    phi = S*V
  end
 
  old_linkdim = dim(commonind(first(psis), phi))

  PH = set_nsite(PH, 2)
  PH = position(PH, psi, verts)

  # don't expand if we are already at maxdim
  (old_linkdim >= maxdim) && return psi, phi0, PH

  linkinds = map(psi -> commonind(psi, phi), psis)

  # compute nullspace to the left and right 
  @timeit_debug timer "nullvector" begin
    nullVecs = map(zip(psis,linkinds)) do (psi,linkind)
      #return nullspace(psi, linkind; atol=atol)
      return implicit_nullspace(psi, linkind)
    end
  end
  
  ##ToDo: Remove since not applicable anymore.
  # if nullspace is empty (happen's for product states with QNs)
  #sum(norm.(nullVecs) .== 0) > 0 && return psi, phi0, PH

  ## build environments
  g = underlying_graph(PH)
  @timeit_debug timer "build environments" begin
    envs = map(zip(verts,psis)) do (n,psi)
      return noprime(mapreduce(*, [v => n for v in neighbors(g, n) if !(v ∈ verts)]; init = psi) do e
        return PH.environments[NamedEdge(e)]
      end *PH.H[n])
    end
  end
  envs=[last(nullVecs)(last(envs)),first(nullVecs)(first(envs))]

  ininds = uniqueinds(last(psis),phi)
  outinds = uniqueinds(first(psis),phi)
  cin=combiner(ininds)
  cout=combiner(outinds)
  envs=[cin*envs[1],cout*envs[2]]
  #@show inds.(envs)
  #@show inds(phi)
  #@show inds(contract([last(envs),phi,first(envs)]))
  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([last(envs),phi,first(envs)], uniqueinds(inds(cout),outinds), uniqueinds(inds(cin),ininds))
  #@show eltype(envMap)
  
  envMapDag = adjoint(envMap)

  @timeit_debug timer "svd_func" begin
    if svd_func==ITensorNetworks._svd_solve_normal
      U,S,V = svd_func(envMap, uniqueinds(inds(cout),outinds); maxdim=maxdim-old_linkdim, cutoff=cutoff)
    elseif svd_func==ITensorNetworks.rsvd_iterative
      #@show theflux
      U,S,V = svd_func(eltype(first(envMap.itensors)),envMap,uniqueinds(inds(cout),outinds);theflux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      use_absolute_cutoff=true)
    else
      U,S,V = svd_func(eltype(envMap),envMap,envMapDag, uniqueinds(inds(cout),outinds); flux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff)
    end
  end
  isnothing(U) && return psi, phi0, PH
  
  U *= dag(cout)
  V *= dag(cin)
  @timeit_debug timer "direct sum" begin
    new_psis = map(zip(psis, [U,V])) do (psi,exp_basis)
      #@show tags(commonind(psi,phi))
      return ITensors.directsum(
        psi => commonind(psi, phi), exp_basis => uniqueind(exp_basis, psi); tags=tags(commonind(psi,phi)),
      )
    end
  end
  new_inds = [last(x)  for x in new_psis]
  new_psis = [first(x) for x in new_psis]
  @assert sum(findmax.(dim.(new_inds))[1] .> maxdim) == 0
  
  phi_indices = replace(inds(phi), (commonind(phi,psis[n]) => dag(new_inds[n]) for n in 1:2)...)
  if hasqns(phi)
    new_phi=ITensor(eltype(phi),flux(phi), phi_indices...)
    fill!(new_phi,0.0)
  else
    new_phi = ITensor(eltype(phi), phi_indices...)
  end

  map(eachindex(phi)) do I
    v = phi[I]
    !iszero(v) && (return new_phi[I]=v)
  end
  ##ToDo:maybe trigger with debug flag
  #old_twosite_tensor = first(psis)*phi*last(psis)

  combiners = map(new_ind -> combiner(new_ind; tags=tags(new_ind), dir=dir(new_ind)), new_inds)
  for (v,new_psi,C) in zip(verts,new_psis,combiners)
    psi[v] = noprime(new_psi*C)
  end

  new_phi = dag(first(combiners)) * new_phi * dag(last(combiners))
  #@show dims(new_phi), dims(phi)
  ##ToDo:maybe trigger with debug flag
  # @assert norm(psi[last(verts)]*new_phi*psi[first(verts)] - old_twosite_tensor) < 50*eps(Float64)
  
  if typeof(region) != NamedEdge{Int}
    psi[n1]*=new_phi
    new_phi=psi[n1]
  end
  return psi, new_phi, PH
  
end

function _full_expand_core_vertex(
    PH, psi, phi, region, svd_func; direction, expand_dir=-1, expander_cache=Any[], maxdim, cutoff, atol=1e-8, kwargs...,
)
  #enforce expand_dir in the tested direction
  @assert expand_dir==-1

  ### only on edges
  #(typeof(region)!=NamedEdge{Int}) && return psi, phi, PH
  ### only on verts
  (typeof(region)==NamedEdge{Int}) && return psi, phi, PH
  ## kind of hacky - only works for mps. More general?
  n1 = first(region)
  theflux=flux(psi[n1])
  #expand_dir=-1
  if direction == 1 
    n2 = expand_dir == 1 ? n1 - 1 : n1+1
  else
    n2 = expand_dir == 1 ? n1 + 1 : n1-1
  end
  (n2 < 1 || n2 > length(vertices(psi))) && return psi,phi,PH
  verts = [n1,n2]
  
  if isempty(expander_cache)
    @warn("building environment of H^2 from scratch!")
  
    g = underlying_graph(PH.H)
    H = vertex_data(data_graph(PH.H))

    H_dag = swapprime.(prime.(dag.(H)), 1,2, tags = "Site")
    H2_vd= replaceprime.(map(*, H, H_dag), 2 => 1)
    H2_ed = edge_data(data_graph(PH.H))

    H2 = TTN(ITensorNetwork(DataGraph(g, H2_vd, H2_ed)), PH.H.ortho_center)
    PH2 = ProjTTN(H2)
    PH2 = set_nsite(PH2, 2)

    push!(expander_cache, PH2)
  end

  PH2 = expander_cache[1]
  n1 = verts[2]
  n2 = verts[1]
  
  ###do we really need to make the environments two-site?
  ###look into how ProjTTN works
  PH2 = position(PH2, psi, [n1,n2])
  PH = position(PH, psi, [n1,n2])

  psi1 = psi[n1]  
  psi2 = psi[n2] #phi
  old_linkdim = dim(commonind(psi1, psi2))

  # don't expand if we are already at maxdim
  ## make more transparent that this is not the normal maxdim arg but maxdim_expand
  ## when would this even happen?
  old_linkdim >= maxdim && return psi, phi, PH

  # compute nullspace to the left and right 
  linkind_l = commonind(psi1, psi2)
  nullVec = implicit_nullspace(psi1, linkind_l)

  # if nullspace is empty (happen's for product states with QNs)
  ## ToDo implement norm or equivalent for a generic LinearMap (i guess via sampling a random vector)
  #norm(nullVec) == 0.0 && return psi, phi, PH


  ## compute both environments
  g = underlying_graph(PH)

  @timeit_debug timer "build environments" begin
    env1 = noprime(mapreduce(*, [v => n1 for v in neighbors(g, n1) if v != n2]; init = psi1) do e
      return PH.environments[NamedEdge(e)]
    end *PH.H[n1]
    )
    env2p = noprime(mapreduce(*, [v => n2 for v in neighbors(g, n2) if v != n1]; init = psi2) do e
      return PH.environments[NamedEdge(e)]
    end *PH.H[n2]
    )
    

    env2 = mapreduce(*, [v => n2 for v in neighbors(g, n2) if v != n1]; init = psi2) do e
      return PH2.environments[NamedEdge(e)]
    end * PH2.H[n2]*prime(dag(psi2))
  end

  env1=nullVec(env1)

  outinds = uniqueinds(psi1,psi2)
  ininds = dag.(outinds)
  cout=combiner(outinds)
  env1 *= cout
  env2p *= prime(dag(psi[n2]),commonind(dag(psi[n2]),dag(psi[n1])))
  env2p2= replaceprime(env2p * replaceprime(dag(env2p),0=>2),2=>1)


  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([prime(dag(env1)),(env2-env2p2),env1] , prime(dag(uniqueinds(cout,outinds))), uniqueinds(cout,outinds))
  #envMap=transpose(envMap)
  #@show typeof(envMap)
  envMapDag=adjoint(envMap)
  # svd-decomposition
  @timeit_debug timer "svd_func" begin
    if svd_func==ITensorNetworks._svd_solve_normal
      U,S,_ = svd_func(envMap, uniqueinds(inds(cout),outinds); maxdim=maxdim-old_linkdim, cutoff=cutoff)
    elseif svd_func==ITensorNetworks.rsvd_iterative
      #@show theflux
      envMap=transpose(envMap)
      U,S,V = svd_func(eltype(first(envMap.itensors)),envMap,ITensors.ITensorNetworkMaps.input_inds(envMap);theflux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      use_absolute_cutoff=true)
      #U,S,V =  svd_func(contract(envMap),uniqueinds(inds(cout),outinds);theflux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      #use_absolute_cutoff=true)
    else
      U,S,_= svd_func(eltype(envMap),envMap,envMapDag,uniqueinds(cout,outinds); flux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff)
    end
  end
  isnothing(U) && return psi,phi,PH

  @assert dim(commonind(U, S)) ≤ maxdim
  #@show inds(U)
  #@show inds(V)
  nullVec = dag(cout)*U
  new_psi1, new_ind1 = ITensors.directsum(
    psi1 => uniqueinds(psi1, nullVec), nullVec => uniqueinds(nullVec, psi1); tags=(tags(commonind(psi1,phi)),)
  )
  new_ind1 = new_ind1[1]
  @assert dim(new_ind1) <= maxdim

  Cl = combiner(new_ind1; tags=tags(new_ind1), dir=dir(new_ind1))

  phi_indices = replace(inds(phi), commonind(phi,psi1) => dag(new_ind1))
 
  if hasqns(phi)
    new_phi=ITensor(eltype(phi),flux(phi), phi_indices...)
    fill!(new_phi,0.0)
  else
    new_phi = ITensor(eltype(phi), phi_indices...)
  end

  map(eachindex(phi)) do I
    v = phi[I]
    !iszero(v) && (return new_phi[I]=v)
  end

  psi[n1] = noprime(new_psi1*Cl)
  new_phi = dag(Cl)*new_phi

  return psi, new_phi, PH
end
