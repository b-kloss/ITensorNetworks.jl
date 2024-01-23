function two_site_expansion_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  (;maxdim,cutoff,time_step) = region_kwargs
 # @show maxdim, cutoff, time_step
 # @show region_kwargs
  
  #ToDo: handle timestep==Inf for DMRG case
  default_updater_kwargs = (;
    svd_func_expand=rsvd_iterative,
    maxdim = Int(ceil((2.0 ^ (1 / 3))) * maxdim),
    cutoff = isinf(time_step) ? cutoff : cutoff/abs(time_step), # ToDo verify that this is the correct way of scaling the cutoff
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)
  #@show updater_kwargs
  # if on edge return without doing anything
  region=first(sweep_plan[which_region_update])
  #@show region, which_region_update == length(sweep_plan)
  typeof(region)<:NamedEdge && return init, (;)
  region=only(region)
  #figure out next region, since we want to expand there
  #ToDo account for non-integer indices into which_region_update
  next_region = which_region_update == length(sweep_plan) ? nothing : first(sweep_plan[which_region_update + 1]) 
  previous_region = which_region_update == 1 ? nothing : first(sweep_plan[which_region_update - 1])
  isnothing(next_region) && return init, (;)
  #@show region, next_region
  #@show typeof(first(sweep_plan[which_region_update])), typeof(region), typeof(next_region), next_region
  !(typeof(next_region)<:NamedEdge) && return init, (;)
  #@assert typeof(next_region)<:NamedEdge
  #@show region, next_region, dst(next_region), src(next_region)
  next_vertex= src(next_region) == region ? dst(next_region) : src(next_region)
  vp = region=>next_vertex
  #dereference and copy # ToDo pass the references into two_site_expand_core 
  PH=copy(projected_operator![])
  state=copy(state![])
  state, phi, PH, has_changed = _two_site_expand_core(PH,state,init,region,vp;  expand_dir=1, updater_kwargs...)
  #rereference, regardless of outcome
  _nsites = (region isa AbstractEdge) ? 0 : length(region) #should be 1
  #@show _nsites
  PH = set_nsite(PH, _nsites)
  PH = position(PH, state, first(sweep_plan[which_region_update]))
  #@show nsite(PH)
  projected_operator![]=PH
  state![]=state
  
  !has_changed && return init, (;)
  
  
  return phi, (;)
end

function implicit_nullspace(A, linkind)
  ###only works when applied in the direction of the environment tensor, not the other way (due to use of ITensorNetworkMap which is not reversible)
  outind=uniqueinds(A,linkind)
  inind=outind  #?
  P=ITensors.ITensorNetworkMaps.ITensorNetworkMap([prime(dag(A),linkind),prime(A,linkind)],inind,outind)
  return x::ITensor -> x-P(x)
end

function _two_site_expand_core(
  PH, psi, phi0, region,vertexpair::Pair;
  expand_dir=1,
  svd_func_expand,
  cutoff,
  maxdim,
  use_relative_cutoff,
  use_absolute_cutoff
) 
  #@show "expanding"
  #@show vertexpair
  theflux=flux(phi0)
  svd_func=svd_func_expand
  v1=first(vertexpair)
  v2=last(vertexpair)
  #@show n1, n2
  verts = [v1,v2]
  n1,n2=1,2
  psis = map(n -> psi[n], verts)  # extract local site tensors
  left_inds = uniqueinds(psis[n1], psis[n2])

  
  U, S, V = svd(psis[n1], left_inds; lefttags=tags(commonind(psis[n1],psis[n2])), righttags=tags(commonind(psis[n1],psis[n2])))
  psis[n1]= U
  phi = S*V
  ##body start

  old_linkdim = dim(commonind(first(psis), phi))

  PH = set_nsite(PH, 2)
  PH = position(PH, psi, verts)

  # don't expand if we are already at maxdim
  (old_linkdim >= maxdim) && return psi, phi0, PH, false
  #@show "expanding", maxdim-old_linkdim
  linkinds = map(psi -> commonind(psi, phi), psis)

  # compute nullspace to the left and right 
  @timeit_debug timer "nullvector" begin
    nullVecs = map(zip(psis,linkinds)) do (psi,linkind)
      #return nullspace(psi, linkind; atol=atol)
      return implicit_nullspace(psi, linkind)
    end
  end
  
  ## build environments
  g = underlying_graph(PH)
  @timeit_debug timer "build environments" begin
    envs = map(zip(verts,psis)) do (n,psi)
      return noprime(mapreduce(*, [v => n for v in neighbors(g, n) if !(v ∈ verts)]; init = psi) do e
        return PH.environments[NamedEdge(e)]
      end *PH.H[n])
    end
  end
  @show inds(first(envs))
  @show inds(last(envs))
  
  envs=[last(nullVecs)(last(envs)),first(nullVecs)(first(envs))]

  ininds = uniqueinds(last(psis),phi)
  outinds = uniqueinds(first(psis),phi)
  cin=combiner(ininds)
  cout=combiner(outinds)
  envs=[cin*envs[1],cout*envs[2]]
  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([last(envs),phi,first(envs)], uniqueinds(inds(cout),outinds), uniqueinds(inds(cin),ininds))
  
  envMapDag = adjoint(envMap)

  @timeit_debug timer "svd_func" begin
    if svd_func==ITensorNetworks._svd_solve_normal
      U,S,V = svd_func(envMap, uniqueinds(inds(cout),outinds); maxdim=maxdim-old_linkdim, cutoff=cutoff)
    elseif svd_func==ITensorNetworks.rsvd_iterative
      U,S,V = svd_func(eltype(first(envMap.itensors)),envMap,uniqueinds(inds(cout),outinds);theflux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      use_absolute_cutoff=true)
    else
      U,S,V = svd_func(eltype(envMap),envMap,envMapDag, uniqueinds(inds(cout),outinds); flux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff)
    end
  end
  isnothing(U) && return psi, phi0, PH, false
  ###FIXME: somehow the svd funcs sometimes return empty ITensors instead of nothing, that should be caught in the SVD routines instead...
  all(isempty.([U,S,V])) && return psi, phi0, PH, false
  
  U *= dag(cout)
  V *= dag(cin)
  @timeit_debug timer "direct sum" begin
    new_psis = map(zip(psis, [U,V])) do (psi,exp_basis)
      
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
  #
  if typeof(region) != NamedEdge{Int}
    psi[v1]=psi[v1]*new_phi
    new_phi=psi[v1]
  end

  return psi, new_phi, PH, true
  ##body end
end


function _full_expand_core_vertex(
    PH, psi, phi, region, svd_func; direction, expand_dir=-1, expander_cache=Any[], maxdim, cutoff, atol=1e-8, kwargs...,
) ###ToDo: adapt to current interface, broken as of now.
  #@show cutoff
  #enforce expand_dir in the tested direction
  @assert expand_dir==-1
  #println("in full expand")
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
  neutralflux=flux(psi[n2])
  
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
  #@show old_linkdim, maxdim
  
  old_linkdim >= maxdim && return psi, phi, PH
  #@show "expandin", maxdim, old_linkdim, maxdim-old_linkdim
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
  envMapDag=adjoint(envMap)
  # svd-decomposition
  @timeit_debug timer "svd_func" begin
    if svd_func==ITensorNetworks._svd_solve_normal
      U,S,_ = svd_func(envMap, uniqueinds(inds(cout),outinds); maxdim=maxdim-old_linkdim, cutoff=cutoff)
    elseif svd_func==ITensorNetworks.rsvd_iterative
      #@show theflux
      envMap=transpose(envMap)
      U,S,_ = svd_func(eltype(first(envMap.itensors)),envMap,ITensors.ITensorNetworkMaps.input_inds(envMap);theflux=neutralflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      use_absolute_cutoff=true)
      #U,S,V =  svd_func(contract(envMap),uniqueinds(inds(cout),outinds);theflux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      #use_absolute_cutoff=true)
    else
      U,S,_= svd_func(eltype(envMap),envMap,envMapDag,uniqueinds(cout,outinds); flux=neutralflux, maxdim=maxdim-old_linkdim, cutoff=cutoff)
    end
  end
  isnothing(U) && return psi,phi,PH
  ###FIXME: somehow the svd funcs sometimes return empty ITensors instead of nothing, that should be caught in the SVD routines instead...
  all(isempty.([U,S])) && return psi, phi, PH
  @assert dim(commonind(U, S)) ≤ maxdim-old_linkdim
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
