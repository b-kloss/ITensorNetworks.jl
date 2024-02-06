function get_column_space(A::Vector{<:ITensor}, lc::Index,rc::Index)
  #gets non-zero blocks in rc by sticking in lc and contracting through
  viable_sectors=Vector{Pair{QN,Int64}}()
  for s in space(lc)
    qn = first(s)
    trial=dag(randomITensor(qn,lc))
    adtrial=foldl(*,A;init=trial)
    nnzblocks(adtrial)==0 && continue
    @assert nnzblocks(adtrial)==1
    i = Int(nzblocks(adtrial)[1])
    thesector=space(only(inds(adtrial)))[i]
    push!(viable_sectors, thesector)
  end
  return viable_sectors
end

function get_column_space(A::ITensor, lc::Index,rc::Index)
  #gets non-zero blocks in rc by sticking in lc and contracting through
  viable_sectors=Vector{Pair{QN,Int64}}()
  ind_loc=only(findall(isequal(rc),inds(A)))
  unique_qns=unique(qns(A,ind_loc))
  for s in space(rc)
    qn = first(s)
    qn in unique_qns || continue 
    push!(viable_sectors, s)
  end
  return viable_sectors
end




function build_guess_matrix(
    eltype::Type{<:Number}, ind, sectors::Union{Nothing,Vector{Pair{QN,Int64}}}, n::Int, p::Int
    )
    if hasqns(ind)
        aux_spaces = Pair{QN,Int64}[]
        for s in sectors
            thedim = last(s)
            qn = first(s)
            en = min(n + p, thedim)
            push!(aux_spaces, Pair(qn, en))
        end
        aux_ind = Index(aux_spaces; dir=dir(ind))
        try
            M = randomITensor(eltype, dag(ind), aux_ind)    #defaults to zero flux
           # @show theflux, flux(M)
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
    #@show M
    return M
end

qns(t::ITensor) = qns(collect(eachnzblock(t)), t)
qns(t::ITensor, i::Int) = qns(collect(eachnzblock(t)), t, i)

function qns(bs::Vector{Block{n}}, t::ITensor) where {n}
  return [qns(b, t) for b in bs]
end

function qns(bs::Vector{Block{n}}, t::ITensor,i::Int) where {n}
  return [qns(b, t)[i] for b in bs]
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
  eltype::Type{<:Number}, ind,  sectors::Union{Nothing,Vector{Pair{QN,Int64}}}, ndict::Dict;) ##given ndict, sectors is not necessary here
  if hasqns(ind)
    #translate_qns = get_qn_dict(ind, theflux; auxdir)
    aux_spaces = Pair{QN,Int64}[]
    #@show first.(space(ind))
    for s in sectors
      thedim = last(s)
      qn = first(s)
      en = min(ndict[qn], thedim)
      push!(aux_spaces, Pair(qn, en))
    end
    aux_ind = Index(aux_spaces; dir=dir(ind))
    #M=randomITensor(eltype,-theflux,dag(ind),aux_ind)
    #@show aux_ind
    try
      M = randomITensor(eltype, dag(ind), aux_ind)
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

function init_guess_sizes(cind, sectors::Union{Nothing,Vector{Pair{QN,Int64}}}, n::Int, rule; theflux=nothing, auxdir=dir(cind))
  if hasqns(cind)
    ndict = Dict{QN,Int64}()
    for s in sectors
      thedim = last(s)
      qn = first(s)
      ndict[qn] = min(rule(n), thedim)
    end
  else
    @assert sectors==nothing
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
  ##deal with non QN tensors first
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
  ##deal with QN tensors second
  ##a lot of this would be more convenient with ITensor internal_inds_space
  ##ToDo: refactor, but not entirely trivial because functionality is implemented on the level of QNIndex, not a set of QNIndices
  ##e.g. it is cumbersome to query the collection of QNs associated with a Block{n} of an ITensor with n>1
  
  oU,oV = old_fact.U, old_fact.V
  nU,nV = new_fact.U, new_fact.V
  nrind=uniqueind(nV,nS)
  orind=uniqueind(oV,oS)
  
  ncrind=commonind(nS,nV)
  ocrind=commonind(nS,nV)
  
  nqns=first.(space(nrind))
  oqns=first.(space(nrind))
  for qn in keys(ndict)
    if !(qn in nqns) &&  !(qn in oqns)
      conv_bool=true
      continue
    #qn
    elseif !(qn in nqns)
      ndict[qn] *= 2
      conv_bool_total = false
      warn("QN was in old factorization but not in new one, this shouldn't happen often!")
      continue
    elseif !(qn in oqns)
      ndict[qn] *= 2
      conv_bool_total = false
      continue
    end
    #qn present in both old and new factorization, grab singular values to compare
    #since U and V are isometries, we can look for the same QN in the central index
    soS=space(ocrind)
    snS=space(ocrind)
    
  #soS = space(inds(oS)[1])
  #snS = space(inds(nS)[1])
  #qns = union(ITensors.qn.(soS), ITensors.qn.(snS))

    oblocks = eachnzblock(oS)
    oblockdict = Int.(getindex.(oblocks, 1))
    oqnindtoblock = Dict(collect(values(oblockdict)) .=> collect(keys(oblockdict)))

    nblocks = eachnzblock(nS)
    nblockdict = Int.(getindex.(nblocks, 1))
    nqnindtoblock = Dict(collect(values(nblockdict)) .=> collect(keys(nblockdict)))

    oqnind = findfirst((first.(soS)) .== [qn])
    nqnind = findfirst((first.(snS)) .== (qn,))
    oblock = oqnindtoblock[oqnind]
    nblock = nqnindtoblock[nqnind]
    ovals = diag(oS[oblock])
    nvals = diag(nS[nblock])
    conv_bool = is_converged_block(collect(ovals), collect(nvals); svd_kwargs...)
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



