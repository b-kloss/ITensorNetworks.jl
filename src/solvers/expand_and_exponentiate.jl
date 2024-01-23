function local_expand_and_exponentiate_updater(
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
  exponentiate_kwargs=updater_kwargs.exponentiate_kwargs
  expand_kwargs=updater_kwargs.expand_kwargs
  #@show inds(init)
  #call expand_updater, no need to define defaults
  expanded_init , _ = two_site_expansion_updater(init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs=expand_kwargs)
  #ToDo: also call exponentiate_updater, instead of reimplementing it here
    # update environment
  #@show inds(expanded_init)
  default_exponentiate_kwargs = (;
    krylovdim=30,
    maxiter=100,
    verbosity=0,
    tol=1E-8,
    ishermitian=true,
    issymmetric=true,
    eager=true,
  )
  exponentiate_kwargs = merge(default_exponentiate_kwargs, exponentiate_kwargs)  #last collection has precedence
  result, exp_info = exponentiate(
    projected_operator![], region_kwargs.time_step, expanded_init; exponentiate_kwargs...
  )
  region=first(sweep_plan[which_region_update])
  typeof(region)<:NamedEdge && return result, (; info=exp_info)
  #truncate
  (;maxdim,cutoff)=region_kwargs
  region=only(region)
  next_region = which_region_update == length(sweep_plan) ? nothing : first(sweep_plan[which_region_update + 1]) 
  previous_region = which_region_update == 1 ? nothing : first(sweep_plan[which_region_update - 1])
  isnothing(next_region) && return result, (; info=exp_info)
  !(typeof(next_region)<:NamedEdge) && return result, (; info=exp_info)
  left_inds = uniqueinds(state![], next_region)
  #ToDo: return truncation error and append to info
  #println("truncating")
  #@show maxdim, cutoff
  U, S, V = svd(result, left_inds; lefttags=tags(state![], next_region), righttags=tags(state![], next_region),maxdim, cutoff)
  next_vertex= src(next_region) == region ? dst(next_region) : src(next_region)
  state=copy(state![])
  #@show inds(V)
  state[next_vertex]=state[next_vertex] * V
  _nsites = (region isa AbstractEdge) ? 0 : length(region) #should be 1
  #@show _nsites
  PH=copy(projected_operator![])
  PH = set_nsite(PH, 2)
  PH = position(PH, state, [region, next_vertex])
  PH = set_nsite(PH, _nsites)
  PH = position(PH, state, first(sweep_plan[which_region_update]))
  state![]=state
  projected_operator![]=PH
 # @show size(S)
  return U*S, (; info=exp_info)
end
