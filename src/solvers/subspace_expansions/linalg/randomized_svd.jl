function get_qn_dict(ind, theflux; auxdir=ITensors.In)
  ###should work right away for multiple QNs, no need to refactor
  @assert hasqns(ind)
  thedir = dir(ind)
  translator = Dict{QN,QN}()
  for s in space(ind)
    qn = first(s)
    translator[qn] = (-auxdir) * ((-thedir) * qn + theflux)
  end
  return translator
end

function build_guess_matrix(
  eltype::Type{<:Number}, ind, n::Int, p::Int; theflux=nothing, auxdir=ITensors.In
)
  if hasqns(ind)
    translate_qns = get_qn_dict(ind, theflux; auxdir)
    aux_spaces = Pair{QN,Int64}[]
    for s in space(ind)
      thedim = last(s)
      qn = first(s)
      en = min(n + p, thedim)
      push!(aux_spaces, Pair(translate_qns[qn], en))
    end
    aux_ind = Index(aux_spaces; dir=auxdir)
    try
      M = randomITensor(eltype, -theflux, dag(ind), aux_ind)
    catch e
      @show e
      @show aux_ind
      @show ind
      error("stopping here something is wrong")
    end
    @assert nnzblocks(M) != 0
  else
    thedim = dim(ind)
    en = min(n + p, thedim)
    #ep = max(0,en-n)
    aux_ind = Index(en)
    M = randomITensor(eltype, ind, aux_ind)
  end
  return M
end
qns(t::ITensor) = qns(collect(eachnzblock(t)), t)
function qns(bs::Vector{Block{n}}, t::ITensor) where {n}
  return [qns(b, t) for b in bs]
end

function qns(b::Block{n}, t::ITensor) where {n}
  theqns = QN[]
  for i in 1:order(t)
    theb = getindex(b, i)
    theind = inds(t)[i]
    push!(theqns, ITensors.qn(space(theind), Int(theb)))
  end
  return theqns
end

function build_guess_matrix(
  eltype::Type{<:Number}, ind, ndict::Dict; theflux=nothing, auxdir=ITensors.In
)
  if hasqns(ind)
    translate_qns = get_qn_dict(ind, theflux; auxdir)
    aux_spaces = Pair{QN,Int64}[]
    #@show first.(space(ind))
    for s in space(ind)
      thedim = last(s)
      qn = first(s)
      en = min(ndict[translate_qns[qn]], thedim)
      push!(aux_spaces, Pair(translate_qns[qn], en))
    end
    aux_ind = Index(aux_spaces; dir=auxdir)
    #M=randomITensor(eltype,-theflux,dag(ind),aux_ind)
    #@show aux_ind
    try
      M = randomITensor(eltype, -theflux, dag(ind), aux_ind)
    catch e
      @show e
      @show aux_ind
      @show ind
      error("stopping here something is wrong")
    end
    @assert nnzblocks(M) != 0
  else
    thedim = dim(ind)
    en = min(ndict[space(ind)], thedim)
    aux_ind = Index(en)
    M = randomITensor(eltype, ind, aux_ind)
  end
  return M
end

function init_guess_sizes(cind, n::Int, rule; theflux=nothing, auxdir=ITensors.In)
  #@show typeof(cind)
  #@show cind
  #@show theflux
  if hasqns(cind)
    translate_qns = get_qn_dict(cind, theflux)
    ndict = Dict{QN,Int64}()
    for s in space(cind)
      thedim = last(s)
      qn = first(s)
      ndict[translate_qns[qn]] = min(rule(n), thedim)
    end
  else
    thedim = dim(cind)
    ndict = Dict{Int64,Int64}()
    ndict[thedim] = min(thedim, rule(n))
  end
  return ndict
end

function increment_guess_sizes(ndict::Dict{QN,Int64}, n_inc::Int, rule)
  for key in keys(ndict)
    #thedim=last(key)
    ndict[key] = ndict[key] + rule(n_inc)
  end
  return ndict
end

function increment_guess_sizes(ndict::Dict{Int64,Int64}, n_inc::Int, rule)
  for key in keys(ndict)
    #thedim=key
    ndict[key] = ndict[key] + rule(n_inc)
  end
  return ndict
end
#=
function increment_guess_sizes(ndict::Dict{QN,Int64},old_fact,new_fact,n_inc::Int;svd_kwargs...)
    oS=old_fact.S
    nS=new_fact.S

    #@show keys(ndict)
    for block in eachnzblock(oS)
        i=Int(block[1])
        #assumes same ordering of inds
        oldspace=space(inds(oS)[1])
        newspace=space(inds(nS)[1])
        #@show oldspace
        #@show newspace
        #make sure blocks are the same QN when we compare them
        @assert first(oldspace[i])==first(newspace[i])
        ovals=diag(oS[block])
        nvals=diag(nS[block])
        conv_bool=is_converged_block(collect(ovals),collect(nvals);svd_kwargs...)
        if !conv_bool
            ndict[first(newspace[i])]+=n_inc   ###this is wrong
        end
    end
    return ndict
end
=#
function approx_the_same(o, n; abs_eps=1e-12, rel_eps=1e-6)
  absdev = abs.(o .- n)
  reldev = abs.((o .- n) ./ n)
  abs_conv = absdev .< abs_eps
  rel_conv = reldev .< rel_eps
  return all(abs_conv .|| rel_conv)
end

function is_converged_block(o, n; svd_kwargs...)
  maxdim = get(svd_kwargs, :maxdim, Inf)
  if length(o) != length(n)
    return false
  else
    r = min(maxdim, length(o))
    #ToDo: Pass kwargs?
    return approx_the_same(o[1:r], n[1:r])
  end
end

function is_converged!(ndict, old_fact, new_fact; n_inc=1, has_qns=true, svd_kwargs...)
  oS = old_fact.S
  nS = new_fact.S
  theflux = flux(nS)
  oldflux = flux(oS)
  if has_qns
    if oldflux == nothing || theflux == nothing
      if norm(oS) == 0.0 && norm(nS) == 0.0
        return true
      else
        return false
      end
    else
      try
        @assert theflux == flux(oS)
      catch e
        @show e
        @show theflux
        @show oldflux
        error("Somehow the fluxes are not matching here! Exiting")
      end
    end
  else
    ###not entirely sure if this is legal for empty factorization
    if norm(oS) == 0.0
      if norm(nS) == 0.0
        return true
      else
        return false
      end
    end
  end

  maxdim = get(svd_kwargs, :maxdim, Inf)
  os = sort(storage(oS); rev=true)
  ns = sort(storage(nS); rev=true)
  if length(os) >= maxdim && length(ns) >= maxdim
    conv_global = approx_the_same(os[1:maxdim], ns[1:maxdim])
  elseif length(os) != length(ns)
    conv_global = false
  else
    r = length(ns)
    conv_global = approx_the_same(os[1:r], ns[1:r])
  end
  if !hasqns(oS)
    if conv_global == false
      #ndict has only one key 
      ndict[first(keys(ndict))] *= 2
    end
    return conv_global
  end
  conv_bool_total = true
  ##a lot of this would be more convenient with ITensor internal_inds_space
  ##ToDo: refactor, but not entirely trivial because functionality is implemented on the level of QNIndex, not a set of QNIndices
  ##e.g. it is cumbersome to query the collection of QNs associated with a Block{n} of an ITensor with n>1
  soS = space(inds(oS)[1])
  snS = space(inds(nS)[1])
  qns = union(ITensors.qn.(soS), ITensors.qn.(snS))

  oblocks = eachnzblock(oS)
  oblockdict = Int.(getindex.(oblocks, 1))
  oqnindtoblock = Dict(collect(values(oblockdict)) .=> collect(keys(oblockdict)))

  nblocks = eachnzblock(nS)
  nblockdict = Int.(getindex.(nblocks, 1))
  nqnindtoblock = Dict(collect(values(nblockdict)) .=> collect(keys(nblockdict)))

  for qn in qns
    if qn in ITensors.qn.(snS) && qn in ITensors.qn.(soS)
      oqnind = findfirst((first.(soS)) .== [qn])
      nqnind = findfirst((first.(snS)) .== (qn,))
      oblock = oqnindtoblock[oqnind]
      nblock = nqnindtoblock[nqnind]
      #oblock=ITensors.block(first,inds(oS)[1],qn)
      #nblock=ITensors.block(first,inds(nS)[1],qn)

      #make sure blocks are the same QN when we compare them
      #@assert first(soS[oqnind])==first(snS[nqnind])#
      ovals = diag(oS[oblock])
      nvals = diag(nS[nblock])
      conv_bool = is_converged_block(collect(ovals), collect(nvals); svd_kwargs...)
    else
      conv_bool = false
    end
    if conv_bool == false
      ndict[qn] *= 2
    end
    conv_bool_total *= conv_bool
  end
  if conv_bool_total == true
    @assert conv_global == true
  else
    if conv_global == true
      println(
        "Subspace expansion, rand. svd: singular vals converged globally, but may not be optimal, doing another iteration",
      )
    end
  end
  return conv_bool_total::Bool
end

function rsvd_iterative(
  T,
  A::ITensors.ITensorNetworkMaps.ITensorNetworkMap,
  linds::Vector{<:Index};
  theflux=nothing,
  svd_kwargs...,
)

  #translate from in/out to l/r logic
  ininds = ITensors.ITensorNetworkMaps.input_inds(A)
  outinds = ITensors.ITensorNetworkMaps.output_inds(A)
  @assert linds == ininds  ##FIXME: this is specific to the way we wrote the subspace expansion, should be fixed in another iteration
  rinds = outinds

  CL = combiner(linds...)
  CR = combiner(rinds...)
  cL = uniqueind(inds(CL), linds)
  cR = uniqueind(inds(CR), rinds)

  l = CL * A.itensors[1]
  r = A.itensors[end] * CR

  if length(A.itensors) !== 2
    AC = ITensors.ITensorNetworkMaps.ITensorNetworkMap(
      [l, A.itensors[2:(length(A.itensors) - 1)]..., r],
      commoninds(l, CL),
      commoninds(r, CR),
    )
  else
    AC = ITensors.ITensorNetworkMaps.ITensorNetworkMap(
      [l, r], commoninds(l, CL), commoninds(r, CR)
    )
  end
  ###this initializer part is still a bit ugly
  n_init = 1
  p_rule(n) = 2 * n
  ndict2 = init_guess_sizes(cR, n_init, p_rule; theflux=theflux)
  ndict = init_guess_sizes(cL, n_init, p_rule; theflux=theflux)
  if hasqns(ininds)
    ndict = merge(ndict, ndict2)
  else
    ndict = ndict2
  end
  M = build_guess_matrix(T, cR, ndict; theflux=theflux)
  fact, Q = rsvd_core(AC, M; svd_kwargs...)
  n_inc = 2
  ndict = increment_guess_sizes(ndict, n_inc, p_rule)
  new_fact = deepcopy(fact)
  while true
    M = build_guess_matrix(T, cR, ndict; theflux=theflux)
    new_fact, Q = rsvd_core(AC, M; svd_kwargs...)
    if is_converged!(ndict, fact, new_fact; n_inc, has_qns=hasqns(ininds), svd_kwargs...)
      break
    else
      fact = new_fact
    end
  end
  vals = diag(array(new_fact.S))
  (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs, :cutoff, 0.0)) &&
    return nothing, nothing, nothing
  return dag(CL) * Q * new_fact.U, new_fact.S, new_fact.V * dag(CR)
end

function rsvd_iterative(A::ITensor, linds::Vector{<:Index}; svd_kwargs...)
  rinds = uniqueinds(A, linds)
  CL = combiner(linds)
  CR = combiner(rinds)
  AC = CL * A * CR
  cL = combinedind(CL)
  cR = combinedind(CR)
  if inds(AC) != (cL, cR)
    AC = permute(AC, cL, cR)
  end
  n_init = 1
  p_rule(n) = 2 * n
  iszero(norm(AC)) && return nothing, nothing, nothing
  #@show flux(AC)
  ndict2 = init_guess_sizes(cR, n_init, p_rule; theflux=hasqns(AC) ? flux(AC) : nothing)
  ndict = init_guess_sizes(cL, n_init, p_rule; theflux=hasqns(AC) ? flux(AC) : nothing)
  ndict = merge(ndict, ndict2)
  M = build_guess_matrix(eltype(AC), cR, ndict; theflux=hasqns(AC) ? flux(AC) : nothing)
  fact, Q = rsvd_core(AC, M; svd_kwargs...)
  n_inc = 1
  ndict = increment_guess_sizes(ndict, n_inc, p_rule)
  new_fact = deepcopy(fact)
  while true
    M = build_guess_matrix(eltype(AC), cR, ndict; theflux=hasqns(AC) ? flux(AC) : nothing)
    new_fact, Q = rsvd_core(AC, M; svd_kwargs...)
    if is_converged!(ndict, fact, new_fact; n_inc, has_qns=hasqns(AC), svd_kwargs...)
      break
    else
      fact = new_fact
    end
  end
  vals = diag(array(new_fact.S))
  (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs, :cutoff, 0.0)) &&
    return nothing, nothing, nothing
  #@show flux(dag(CL)*Q*new_fact.U)
  #@show flux(new_fact.S)
  @assert flux(new_fact.S) == flux(AC)
  return dag(CL) * Q * new_fact.U, new_fact.S, new_fact.V * dag(CR)
  #ToDo?: handle non-QN case separately because there it is advisable to start with n_init closer to target maxdim_expand
  ##not really an issue anymore since we do *2 increase, so only log number of calls
end

function rsvd(A::ITensor, linds::Vector{<:Index}, n::Int, p::Int; svd_kwargs...)
  rinds = uniqueinds(A, linds)
  #ToDo handle empty rinds
  #boilerplate matricization of tensor for matrix decomp
  CL = combiner(linds)
  CR = combiner(rinds)
  AC = CL * A * CR
  cL = combinedind(CL)
  cR = combinedind(CR)
  if inds(AC) != (cL, cR)
    AC = permute(AC, cL, cR)
  end
  M = build_guess_matrix(eltype(AC), cR, n, p; theflux=hasqns(AC) ? flux(AC) : nothing)
  fact, Q = rsvd_core(AC, M; svd_kwargs)
  return dag(CL) * Q * fact.U, fact.S, fact.V * dag(CR)
end

function rsvd_core(AC::ITensor, M; svd_kwargs...)
  Q = AC * M
  #@show dims(Q)
  Q = ITensors.qr(Q, commoninds(AC, Q))[1]
  QAC = dag(Q) * AC
  fact = svd(QAC, uniqueind(QAC, AC); svd_kwargs...)
  return fact, Q
end

function rsvd_core(AC::ITensors.ITensorNetworkMaps.ITensorNetworkMap, M; svd_kwargs...)
  #assumes that we want to do a contraction of M with map over its maps output_inds, i.e. a right-multiply
  #thus a transpose is necessary
  Q = transpose(AC) * M
  Q = ITensors.qr(Q, ITensors.ITensorNetworkMaps.input_inds(AC))[1]
  QAC = AC * dag(Q)
  @assert typeof(QAC) <: ITensor
  #@show inds(QAC)
  #@assert !iszero(norm(QAC))

  fact = svd(
    QAC, uniqueind(inds(Q), ITensors.ITensorNetworkMaps.input_inds(AC)); svd_kwargs...
  )
  return fact, Q
end

##these are also not sane defaults if we want to keep the expansion at single site cost
function _svd_solve_randomized_precontract(T, envMap, left_ind; maxdim, cutoff, flux)
  return _svd_solve_randomized_precontract(envMap, left_ind, maxdim, maxdim; maxdim, cutoff)
end
###FIXME: override needs to be removed later, when rsvd for Map/Vector of ITensors is implemented
#_svd_solve_randomized_precontract(T,envMap, left_ind;flux, maxdim, cutoff) = _svd_solve_randomized_precontract(T,envMap, left_ind ; maxdim, cutoff) 
function _svd_solve_randomized_precontract(
  T, envMap, envMapDag, left_ind; flux, maxdim, cutoff
)
  return _svd_solve_randomized_precontract(T, envMap, left_ind; maxdim, cutoff, flux)
end
