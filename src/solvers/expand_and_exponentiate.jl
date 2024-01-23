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
  #ToDo: compress with cutoff like it would be done in the alternative implementation of extract_local_tensor
  return result, (; info=exp_info)
end
