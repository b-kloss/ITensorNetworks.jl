function update_sweep(nsite, graph::AbstractGraph; kwargs...)
  return vcat(
    [
      half_sweep(
        direction(half),
        graph,
        make_region;
        nsite,
        region_args=(; half_sweep=half),
        kwargs...,
      ) for half in 1:2
    ]...,
  )
end

function step_expand(
  step_expander,
  PH,
  psi::AbstractTTN;
  cutoff::AbstractFloat=1E-16,
  maxdim::Int=typemax(Int),
  maxdim_expand::Int,
  mindim::Int=1,
  normalize::Bool=false,
  nsite::Int=2,
  outputlevel::Int=0,
  sw::Int=1,
  sweep_regions=update_sweep(nsite, psi),
  kwargs...,
)
  # (Needed to handle user-provided sweep_regions)
  sweep_regions = append_missing_namedtuple.(to_tuple.(sweep_regions))

  if nv(psi) == 1
    error(
      "`alternating_update` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  for (n, (region, step_kwargs)) in enumerate(sweep_regions)
    direction = get(step_kwargs, :substep, 1)
    svd_func = get(kwargs, :svd_func, _svd_solve_normal)

    psi = orthogonalize(psi, current_ortho(region))
    
    
    nsites = (region isa AbstractEdge) ? 0 : length(region)
    #@show region
    #@show current_ortho(region)
    #@show nsites
    PH = set_nsite(PH, nsites)
    if nsites>0
      PH = position(PH, psi, region)
    end
    direction == 1 && continue
    psi, phi = extract_local_tensor(psi, region, maxdim)

    psi,phi,PH = step_expander(
      PH,
      psi,
      phi,
      region,
      svd_func;
      direction=direction,
      maxdim = maxdim_expand,
      expand_dir=-1,
      cutoff,
      kwargs...,
    )

    normalize && (phi /= norm(phi))

    drho = nothing
    ortho = "left"

    (typeof(region)==NamedEdge{Int}) && (PH = position(PH,psi,[src(region),dst(region)]))
    psi, spec = insert_local_tensor(
      psi, phi, region, maxdim; eigen_perturbation=drho, ortho, normalize, kwargs...
    )
    (typeof(region)==NamedEdge{Int}) && (PH = position(PH,psi,region))
  end

  normalize && normalize!(psi)
  return psi, PH
end

function update_step(
  solver,
  PH,
  psi::AbstractTTN;
  cutoff::AbstractFloat=1E-16,
  maxdim::Int=typemax(Int),
  maxdim_expand::Int,
  mindim::Int=1,
  normalize::Bool=false,
  nsite::Int=2,
  outputlevel::Int=0,
  sw::Int=1,
  sweep_regions=update_sweep(nsite, psi),
  kwargs...,
)
  info = nothing
  PH = copy(PH)
  psi = copy(psi)

  observer = get(kwargs, :observer!, nothing)

  # Append empty namedtuple to each element if not already present
  # (Needed to handle user-provided sweep_regions)
  sweep_regions = append_missing_namedtuple.(to_tuple.(sweep_regions))

  if nv(psi) == 1
    error(
      "`alternating_update` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  maxtruncerr = 0.0
  info = nothing
  for (n, (region, step_kwargs)) in enumerate(sweep_regions)
    psi, PH, spec, info = local_update(
      solver,
      PH,
      psi,
      region;
      outputlevel,
      cutoff,
      maxdim,
      maxdim_expand,
      mindim,
      normalize,
      step_kwargs,
      kwargs...,
    )
    maxtruncerr = isnothing(spec) ? maxtruncerr : max(maxtruncerr, spec.truncerr)

    if outputlevel >= 2
      #if get(data(sweep_step),:time_direction,0) == +1
      #  @printf("Sweep %d, direction %s, position (%s,) \n", sw, direction, pos(step))
      #end
      print("  Truncated using")
      @printf(" cutoff=%.1E", cutoff)
      @printf(" maxdim=%.1E", maxdim)
      print(" mindim=", mindim)
      #print(" current_time=", round(current_time; digits=3))
      println()
      # if spec != nothing
      if typeof(pos(sweep_step)) == NamedEdge{Int}
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n",
          spec.truncerr,
          0,
          linkdim(psi, edgetype(psi)(pos(sweep_step)))
        )
      end
      flush(stdout)
    end
    update!(
      observer;
      sweep_step=n,
      total_sweep_steps=length(sweep_regions),
      end_of_sweep=(n == length(sweep_regions)),
      psi,
      region,
      sweep=sw,
      PH,
      spec,
      outputlevel,
      info,
      step_kwargs...,
    )
  end
  # Just to be sure:
  normalize && normalize!(psi)
  return psi, PH, (; maxtruncerr)
end

# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.
#
# In the simplest case, exact_local_tensor contracts together a few
# tensors of the network and returns the result, while 
# insert_local_tensors takes that tensor and factorizes it back
# apart and puts it back into the network.

function extract_local_tensor(psi::AbstractTTN, pos::Vector)
  return psi, prod(psi[v] for v in pos)
end

function extract_local_tensor(psi::AbstractTTN, pos::Vector, m::Int)
  extract_local_tensor(psi, pos)
end

function extract_local_tensor(psi::AbstractTTN, pos::Vector, c::Float64, m::Int)
  extract_local_tensor(psi, pos)
end

function extract_local_tensor(psi::AbstractTTN, e::NamedEdge)
  left_inds = uniqueinds(psi, e)
  U, S, V = svd(psi[src(e)], left_inds; lefttags=tags(psi, e), righttags=tags(psi, e))
  psi[src(e)] = U
  return psi, S * V
end

function extract_local_tensor(psi::AbstractTTN, e::NamedEdge, m::Int)
  left_inds = uniqueinds(psi, e)
  U, S, V = svd(psi[src(e)], left_inds; lefttags=tags(psi, e), righttags=tags(psi, e), maxdim=m)
  psi[src(e)] = U

  return psi, S * V
end

function extract_local_tensor(psi::AbstractTTN, e::NamedEdge, cutoff::Float64,m::Int)
  left_inds = uniqueinds(psi, e)
  U, S, V = svd(psi[src(e)], left_inds; lefttags=tags(psi, e), righttags=tags(psi, e), cutoff=cutoff, maxdim=m)
  psi[src(e)] = U

  return psi, S * V
end

# sort of multi-site replacebond!; TODO: use dense TTN constructor instead
function insert_local_tensor(
  psi::AbstractTTN,
  phi::ITensor,
  pos::Vector,
  maxdim::Int;
  which_decomp=nothing,
  normalize=false,
  eigen_perturbation=nothing,
  kwargs...,
)
  spec = nothing
  for (v, vnext) in IterTools.partition(pos, 2, 1)
    e = edgetype(psi)(v, vnext)
    indsTe = inds(psi[v])
    L, phi, spec = factorize(
      phi, indsTe; which_decomp, tags=tags(psi, e), eigen_perturbation, maxdim=maxdim, kwargs...
    )
    psi[v] = L
    eigen_perturbation = nothing # TODO: fix this
  end
  psi[last(pos)] = phi
  psi = set_ortho_center(psi, [last(pos)])
  @assert isortho(psi) && only(ortho_center(psi)) == last(pos)
  normalize && (psi[last(pos)] ./= norm(psi[last(pos)]))
  # TODO: return maxtruncerr, will not be correct in cases where insertion executes multiple factorizations
  return psi, spec
end

function insert_local_tensor(psi::AbstractTTN, phi::ITensor, e::NamedEdge, maxdim::Int; kwargs...)
  # if dim(commonind(phi, psi[src(e)])) > maxdim
  #   U, S, V = svd(phi, commonind(phi, psi[src(e)]); maxdim, lefttags=tags(commonind(phi,psi[src(e)])), righttags=tags(commonind(phi,psi[src(e)])))
  #   psi[src(e)] *= U
  #   psi[dst(e)] *= S*V
  # else
    psi[dst(e)] *= phi
  # end

  psi = set_ortho_center(psi, [dst(e)])
  return psi, nothing
end

#TODO: clean this up:
current_ortho(::Type{<:Vector{<:V}}, st) where {V} = first(st)
current_ortho(::Type{NamedEdge{V}}, st) where {V} = src(st)
current_ortho(st) = current_ortho(typeof(st), st)

function local_update(
  solver, PH, psi, region; outputlevel, cutoff, maxdim, maxdim_expand=maxdim, mindim, normalize, step_kwargs=NamedTuple(), kwargs...
)
  direction = get(step_kwargs, :substep, 1)
  dt = get(step_kwargs, :time_step,1)
  expander = get(kwargs, :expander, nothing)
  cutoff_expand = get(kwargs, :cutoff_expand,cutoff/abs(dt)^2)
  #@show cutoff_expand
  psi = orthogonalize(psi, current_ortho(region))
  psi, phi = extract_local_tensor(psi, region,  cutoff, maxdim)

  nsites = (region isa AbstractEdge) ? 0 : length(region)
  PH = set_nsite(PH, nsites)
  PH = position(PH, psi, region)

  if !isnothing(expander)
    svd_func = get(kwargs, :svd_func, _svd_solve_normal)
    @timeit_debug timer "local expansion" begin
      psi,phi,PH = expander(
        PH,
        psi,
        phi,
        region,
        svd_func;
        direction,
        maxdim = maxdim_expand,
        cutoff = cutoff_expand,
        kwargs...,
      )
    end

    # update environment
    nsites = (region isa AbstractEdge) ? 0 : length(region)
    PH = set_nsite(PH, nsites)
    PH = position(PH, psi, region)
  end


  info = []
  ### solver behaves weirdly sometimes, stating that PH is not hermitian; this fixes it ###
  #if false
    @timeit_debug timer "local solve" begin
      while true
        try 
          phi, info = solver(PH, phi; normalize, region, step_kwargs..., kwargs...)
          break
        catch e
          @show e
        end
      end
    end
  #end
  normalize && (phi /= norm(phi))

  drho = nothing
  ortho = "left"

  (typeof(region)==NamedEdge{Int}) && (PH = position(PH,psi,[src(region),dst(region)]))
  psi, spec = insert_local_tensor(
    psi, phi, region, maxdim; eigen_perturbation=drho, ortho, normalize, kwargs...
  )
  (typeof(region)==NamedEdge{Int}) && (PH = position(PH,psi,region))

  return psi, PH, spec, info
end
