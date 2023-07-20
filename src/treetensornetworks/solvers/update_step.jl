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

function update_step(
  solver,
  expander,
  PH,
  psi::AbstractTTN;
  cutoff::AbstractFloat=1E-16,
  maxdim::Int=typemax(Int),
  maxdim_expand::Int=maxdim,
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
    psi, PH, spec, info = local_expansion(
      solver,
      expander,
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
  psi[dst(e)] *= phi
  psi = set_ortho_center(psi, [dst(e)])
  return psi, nothing
end

#TODO: clean this up:
current_ortho(::Type{<:Vector{<:V}}, st) where {V} = first(st)
current_ortho(::Type{NamedEdge{V}}, st) where {V} = src(st)
current_ortho(st) = current_ortho(typeof(st), st)

function local_expansion(
  solver, expander, PH, psi, region; outputlevel, cutoff, maxdim, maxdim_expand=maxdim, mindim, normalize, step_kwargs=NamedTuple(), kwargs...
)
  direction = get(step_kwargs, :substep, 1)
  psi = orthogonalize(psi, current_ortho(region))
  psi, phi = extract_local_tensor(psi, region, maxdim)

  nsites = (region isa AbstractEdge) ? 0 : length(region)
  PH = set_nsite(PH, nsites)
  PH = position(PH, psi, region)

  psi,phi,PH = expander(
    PH,
    psi,
    phi,
    region,
    direction;
    maxdim = maxdim_expand,
    cutoff,
    kwargs...,
  )

  phi, PH, info = local_update(
    solver,
    PH,
    phi,
    region;
    outputlevel,
    cutoff,
    maxdim,
    mindim,
    normalize,
    step_kwargs,
    kwargs...,
  )

  # if (typeof(region) != NamedEdge{Int})
  #   n1 = region[1]
  #   n2 = direction == 1 ? n1+1 : n1-1
  #
  #   if n2 <= 16 && n2 > 0
  #     m = 40
  #     if dim(commonind(psi[n2],phi)) > m
  #       PH = position(PH, psi, [n1,n2])
  #       U,S,V = svd(phi*psi[n2], uniqueinds(phi,psi[n2]); maxdim=m, cutoff=1e-12, 
  #                   lefttags = tags(commonind(psi[n2],phi)), righttags = tags(commonind(psi[n2],phi)),
  #                   # use_relative_cutoff=false, use_absolute_cutoff=true, kwargs...,
  #                  )
  #
  #       # @assert psi[n2] * phi ≈ U*S*V
  #       psi[n2] = V
  #       phi = U*S
  #
  #       PH = position(PH, psi, region)
  #     end
  #   end
  # end

  
  # if (typeof(region) == NamedEdge{Int})
  #   (n1,n2) = (src(region),dst(region))
  #   m = 20
  #   if dim(commonind(psi[n1],phi)) > m
  #     # @show dim(commonind(psi[n1],phi))
  #     # @show dim(commonind(psi[n2],phi))
  #     # @show n1,n2
  #     PH = position(PH, psi, [n1,n2])
  #     U,S,V = svd(psi[n1]*phi*psi[n2], uniqueinds(psi[n1],phi); maxdim=m, cutoff=0, 
  #                 lefttags = tags(commonind(psi[n1],phi)), righttags = tags(commonind(psi[n2],phi)),
  #                 # use_relative_cutoff=false, use_absolute_cutoff=true, 
  #                 kwargs...,
  #                )
  #
  #     # @show dim(commonind(U,S))
  #     # @show dim(commonind(V,S))
  #     psi[n1] = U
  #     psi[n2] = V
  #     phi = S
  #
  #     PH = position(PH, psi, region)
  #   end
  # end
  # if (typeof(region) == NamedEdge{Int})
  #   (n1,n2) = (src(region),dst(region))
  #   # @show dim(commonind(psi[n1],phi))
  #   # @show dim(commonind(psi[n2],phi))
  # end
  
  normalize && (phi /= norm(phi))

  drho = nothing
  ortho = "left"

  psi, spec = insert_local_tensor(
    psi, phi, region, maxdim; eigen_perturbation=drho, ortho, normalize, kwargs...
  )

  return psi, PH, spec, info
end

function local_update(
  solver, PH, phi, region; normalize, noise, step_kwargs=NamedTuple(), kwargs...
)
  info = []
  ### solver behaves weirdly sometimes, stating that PH is not hermitian; this fixes it ###
  while true
    try 
      phi, info = solver(PH, phi; normalize, region, step_kwargs..., kwargs...)
      break
    catch e
      @show e
    end
  end

  normalize && (phi /= norm(phi))

  return phi, PH, info
end
