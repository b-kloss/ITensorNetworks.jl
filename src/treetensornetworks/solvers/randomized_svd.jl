function get_qn_dict(ind,theflux;auxdir=ITensors.In)
    ###should work right away for multiple QNs, no need to refactor
    @assert hasqns(ind)
    thedir=dir(ind)
    translator=Dict{QN,QN}()
    for s in space(ind)
        qn = first(s)
        translator[qn]=(-auxdir) * ((-thedir)*qn + theflux )
    end
    return translator
end

function build_guess_matrix(eltype::Type{<:Number},ind,n::Int,p::Int;theflux=nothing,auxdir=ITensors.In)
    if hasqns(ind)
        translate_qns=get_qn_dict(ind,theflux;auxdir)
        aux_spaces=Pair{QN,Int64}[]
        for s in space(ind)
            thedim = last(s)
            qn = first(s)
            en = min(n+p,thedim)
            push!(aux_spaces,Pair(translate_qns[qn],en))
        end
        aux_ind = Index(aux_spaces,dir=auxdir)
        M=randomITensor(eltype,-theflux,dag(ind),aux_ind)
        @assert nnzblocks(M)!=0
    else
        thedim=dim(ind)
        en = min(n+p,thedim)
        #ep = max(0,en-n)
        aux_ind=Index(en)
        M = randomITensor(eltype,ind,aux_ind)
    end
    return M
end

function build_guess_matrix(eltype::Type{<:Number},ind,ndict::Dict;theflux=nothing,auxdir=ITensors.In)
    if hasqns(ind)
        translate_qns=get_qn_dict(ind,theflux;auxdir)
        aux_spaces=Pair{QN,Int64}[]
        #@show first.(space(ind))
        for s in space(ind)
            thedim = last(s)
            qn = first(s)
            en = min(ndict[translate_qns[qn]],thedim)
            push!(aux_spaces,Pair(translate_qns[qn],en))
        end
        aux_ind = Index(aux_spaces,dir=auxdir)
        M=randomITensor(eltype,-theflux,dag(ind),aux_ind)
        @assert nnzblocks(M)!=0
    else
        thedim=dim(ind)
        en = min(ndict[space(ind)],thedim)
        aux_ind=Index(en)
        M = randomITensor(eltype,ind,aux_ind)
    end
    return M

end

function init_guess_sizes(cind,n::Int,rule;theflux=nothing,auxdir=ITensors.in)
    #@show typeof(cind)
    #@show cind
    #@show theflux
    if hasqns(cind)
        translate_qns=get_qn_dict(cind,theflux)
        ndict=Dict{QN,Int64}()
        for s in space(cind)
            thedim = last(s)
            qn = first(s)
            ndict[translate_qns[qn]]=min(rule(n),thedim)            
        end
    else
        thedim=dim(cind)
        ndict=Dict{Int64,Int64}()
        ndict[thedim]=min(thedim,rule(n))
    end
    return ndict
end

function increment_guess_sizes(ndict::Dict{QN,Int64},n_inc::Int,rule)
    for key in keys(ndict)
        #thedim=last(key)
        ndict[key]=ndict[key]+rule(n_inc)
    end
    return ndict
end

function increment_guess_sizes(ndict::Dict{Int64,Int64},n_inc::Int,rule)
    for key in keys(ndict)
        #thedim=key
        ndict[key]=ndict[key]+rule(n_inc)
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
function approx_the_same(o,n;abs_eps=1e-8,rel_eps=1e-4)
    absdev=abs.(o .- n)
    reldev=abs.((o .- n) ./ n)
    abs_conv = absdev .< abs_eps
    rel_conv = reldev .< rel_eps
    return all(abs_conv .|| rel_conv)
end

function is_converged_block(o,n;svd_kwargs...)
    maxdim=get(svd_kwargs, :maxdim,Inf)
    if length(o)!=length(n)
        return false
    else
        r=min(maxdim, length(o))
        #ToDo: Pass kwargs?
        return approx_the_same(o[1:r],n[1:r])
    end
end

function is_converged!(ndict,old_fact,new_fact;n_inc=1,svd_kwargs...)
    #@show svd_kwargs
    oS=old_fact.S
    nS=new_fact.S
    theflux=flux(nS)
    @assert theflux==flux(oS)
    maxdim=get(svd_kwargs, :maxdim,Inf)
    os=sort(storage(oS),rev=true)
    ns=sort(storage(nS),rev=true)
    if length(os)>=maxdim && length(ns) >= maxdim
        conv_global=approx_the_same(os[1:maxdim],ns[1:maxdim])
    elseif length(os)!=length(ns)
        conv_global=false
    else
        r=length(ns)
        conv_global=approx_the_same(os[1:r],ns[1:r])
    end
    if !hasqns(oS)
        if conv_global==false
            #ndict has only one key 
            ndict[first(keys(ndict))]*=2
        end
        return conv_global
    end
    conv_bool_total=true
    ##a lot of this would be more convenient with ITensor internal_inds_space
    ##ToDo: refactor, but not entirely trivial because functionality is implemented on the level of QNIndex, not a set of QNIndices
    ##e.g. it is cumbersome to query the collection of QNs associated with a Block{n} of an ITensor with n>1
    soS=space(inds(oS)[1])
    snS=space(inds(nS)[1])
    qns=union(ITensors.qn.(soS),ITensors.qn.(snS))

    
    oblocks=eachnzblock(oS)
    oblockdict=Int.(getindex.(oblocks,1))
    oqnindtoblock=Dict(collect(values(oblockdict)) .=> collect(keys(oblockdict)))

    nblocks=eachnzblock(nS)
    nblockdict=Int.(getindex.(nblocks,1))
    nqnindtoblock=Dict(collect(values(nblockdict)) .=> collect(keys(nblockdict)))       
    
    for qn in qns
        if qn in ITensors.qn.(snS) && qn in ITensors.qn.(soS)
            oqnind=findfirst((first.(soS)) .== [qn,])
            nqnind=findfirst((first.(snS)) .== (qn,))
            oblock=oqnindtoblock[oqnind]
            nblock=nqnindtoblock[nqnind]
            #oblock=ITensors.block(first,inds(oS)[1],qn)
            #nblock=ITensors.block(first,inds(nS)[1],qn)
            
            #make sure blocks are the same QN when we compare them
            #@assert first(soS[oqnind])==first(snS[nqnind])#
            ovals=diag(oS[oblock])
            nvals=diag(nS[nblock])
            #@show typeof(collect(ovals)),collect(nvals)
            conv_bool=is_converged_block(collect(ovals),collect(nvals);svd_kwargs...)
    #        @show conv_bool
        else
            conv_bool=false
        end
        if conv_bool==false
            ndict[qn]*=2 
        end
        conv_bool_total*=conv_bool
    end
    if conv_bool_total==true
        @assert conv_global==true
    else
        if conv_global==true
            println("svals converged globally, but may not be optimal, doing another iteration")
        end
    end
    return conv_bool_total::Bool
end

function rsvd_iterative(T,A::ITensors.ITensorNetworkMaps.ITensorNetworkMap,linds::Vector{<:Index};theflux=nothing,svd_kwargs...)
    #@show theflux
    #@show inds(contract(A))
    #translate from in/out to l/r logic
    ininds=ITensors.ITensorNetworkMaps.input_inds(A)
    outinds=ITensors.ITensorNetworkMaps.output_inds(A)
    #@show inds(contract(A))
    #@show ininds
    #@show outinds
    @assert linds==ininds  ##FIXME: this is specific to the way we wrote the subspace expansion, should be fixed in another iteration
    rinds=outinds
    ##for subspace expansion we already have combined indices, but combining twice shouldn't incur overhead
    CL=combiner(linds...)
    CR=combiner(rinds...)
    cL=uniqueind(inds(CL),linds)
    cR=uniqueind(inds(CR),rinds)
    #@show inds(CL)
    #@show inds(A.itensors[1])
    #@show inds(CR)
    #@show inds(A.itensors[end])
    
    l=CL*A.itensors[1]
    r=A.itensors[end]*CR
    #@show inds(l)
    #@show inds(r)
    if length(A.itensors)!==2
        AC=ITensors.ITensorNetworkMaps.ITensorNetworkMap([l,A.itensors[2:length(A.itensors)-1]...,r],commoninds(l,CL),commoninds(r,CR))
    else
        AC=ITensors.ITensorNetworkMaps.ITensorNetworkMap([l,r],commoninds(l,CL),commoninds(r,CR))
    end
    #@show inds(AC.itensors[1])
    #@show inds(AC.itensors[2])
    #@show inds(AC.itensors[3])
    
    #no permute necessary
    n_init=1
    p_rule(n)=2*n
    ndict2=init_guess_sizes(cR,n_init,p_rule;theflux=theflux)
    ndict=init_guess_sizes(cL,n_init,p_rule;theflux=theflux)
    if hasqns(ininds)
        ndict=merge(ndict,ndict2)
    else
        ndict=ndict2
    end
    M=build_guess_matrix(T,cR,ndict;theflux=theflux)
    fact,Q=rsvd_core(AC,M;svd_kwargs...)
    n_inc=2
    ndict=increment_guess_sizes(ndict,n_inc,p_rule)
    new_fact=deepcopy(fact)
    #Q=0.0
    while true
        #@show "here"
        M=build_guess_matrix(T,cR,ndict;theflux=theflux)
        new_fact,Q=rsvd_core(AC,M;svd_kwargs...)
        #@show length(storage(new_fact.S)),minimum(new_fact.S),maximum(new_fact.S)
        #ndict=increment_guess_sizes(ndict,n_inc,p_rule)
        #@show ndict
        if is_converged!(ndict,fact,new_fact;n_inc,svd_kwargs...)
            break
        else
            #ndict=increment_guess_sizes(ndict,fact,new_fact,p_rule(n_inc);svd_kwargs...)
            fact=new_fact
            #compare convergence
        end
    end
    #@show new_fact
    #@show minimum(collect(storage(new_fact.S)))
    #@show length(collect(storage(new_fact.S)))
    vals=diag(array(new_fact.S))
    (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs,:cutoff,0.0)) && return nothing,nothing,nothing
    return dag(CL)*Q*new_fact.U, new_fact.S, new_fact.V*dag(CR)
end


function rsvd_iterative(A::ITensor,linds::Vector{<:Index};svd_kwargs...)
    ##preprocess
    #println("in rsvdit")
    rinds=uniqueinds(A,linds)
    #@show svd_kwargs
    #ToDo handle empty rinds
    #boilerplate matricization of tensor for matrix decomp
    CL=combiner(linds)
    CR=combiner(rinds)
    AC=CL*A*CR
    cL=combinedind(CL)
    cR=combinedind(CR)
    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end
    ##build ndict for initial run
    n_init=1
    p_rule(n)=2*n
    #@show hasqns(cR),hasqns(cL)
    ndict2=init_guess_sizes(cR,n_init,p_rule;theflux=hasqns(AC) ? flux(AC) : nothing)
    ndict=init_guess_sizes(cL,n_init,p_rule;theflux=hasqns(AC) ? flux(AC) : nothing)
    ndict=merge(ndict,ndict2)
    #@show ndict
    #@show keys(ndict2)
    #println("in rsvdit - before first guess")
    M=build_guess_matrix(eltype(AC),cR,ndict;theflux=hasqns(AC) ? flux(AC) : nothing)
    fact,Q=rsvd_core(AC,M;svd_kwargs...)

    #println("in rsvdit - after first solve")
    n_inc=1
    ndict=increment_guess_sizes(ndict,n_inc,p_rule)
    new_fact=deepcopy(fact)
    #Q=0.0
    while true
        #@show "here"
        M=build_guess_matrix(eltype(AC),cR,ndict;theflux=hasqns(AC) ? flux(AC) : nothing)
        new_fact,Q=rsvd_core(AC,M;svd_kwargs...)
        #@show type(new_fact)
        #ndict=increment_guess_sizes(ndict,n_inc,p_rule)
        if is_converged!(ndict,fact,new_fact;n_inc,svd_kwargs...)
            break
        else
            #ndict=increment_guess_sizes(ndict,fact,new_fact,p_rule(n_inc);svd_kwargs...)
            fact=new_fact
            #compare convergence
        end
    end
    #@show new_fact
    #@show minimum(collect(storage(new_fact.S)))
    #@show length(collect(storage(new_fact.S)))
    vals=diag(array(new_fact.S))
    (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs,:cutoff,0.0)) && return nothing,nothing,nothing
    return dag(CL)*Q*new_fact.U, new_fact.S, new_fact.V*dag(CR)
    #ToDo: handle non-QN case separately because there it is advisable to start with n_init closer to target maxdim
    
    

end

function rsvd(A::ITensor,linds::Vector{<:Index},n::Int,p::Int;svd_kwargs...)
    rinds=uniqueinds(A,linds)
    #ToDo handle empty rinds
    #boilerplate matricization of tensor for matrix decomp
    CL=combiner(linds)
    CR=combiner(rinds)
    AC=CL*A*CR
    cL=combinedind(CL)
    cR=combinedind(CR)
    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end
    M=build_guess_matrix(eltype(AC),cR,n,p;theflux=hasqns(AC) ? flux(AC) : nothing)
    fact,Q=rsvd_core(AC,M;svd_kwargs)
    return dag(CL)*Q*fact.U, fact.S, fact.V*dag(CR)
    #return ITensors.TruncSVD(dag(CL)*Q*fact.U,fact.S,fact.V*dag(CR),fact.spec,linds,rinds)  #?
end

function rsvd_core(AC::ITensor,M;svd_kwargs...)
    Q=AC*M
    Q=ITensors.qr(Q,commoninds(AC,Q))[1]
    QAC=dag(Q)*AC
    #@show svd_kwargs
    fact=svd(QAC,uniqueind(QAC,AC);svd_kwargs...)
    return fact,Q
end

function rsvd_core(AC::ITensors.ITensorNetworkMaps.ITensorNetworkMap,M;svd_kwargs...)
    #assumes that we want to do a contraction of M with map over its maps output_inds, i.e. a right-multiply
    #thus a transpose is necessary
    #@show inds(contract(AC))
    #@show inds.(AC.itensors)
    Q=transpose(AC)*M
    #@show inds(Q)
    #commoninds(contract(AC),Q) => input_inds(AC)
    Q=ITensors.qr(Q,ITensors.ITensorNetworkMaps.input_inds(AC))[1]
    #left multiply for map looks like this
    QAC=AC*dag(Q)
    #now QAC should be an ITensor
    #println("done with contraction inside core")
    @assert typeof(QAC)<:ITensor
    #@show svd_kwargs
    #@show inds(QAC)
    fact=svd(QAC,uniqueind(inds(Q),ITensors.ITensorNetworkMaps.input_inds(AC));svd_kwargs...)
    return fact,Q
end


function _svd_solve_randomized_precontract(
    envMap, left_ind, n::Int,p::Int; maxdim, cutoff,flux=nothing
  )
    M = ITensors.ITensorNetworkMaps.contract(envMap)
    ##ideally, n and p would be determined by some logic so that we can have the same interface for all svds
    ##user should be able to override n,p via some kwarg
    ##potentially pass a function as kwarg that handles the logic of computing n,p for each sector etc.

    norm(M) ≤ eps(Float64) && return nothing,nothing,nothing
  
    U,S,V = rsvd(
      M,
      left_ind,
      n,
      p;
      maxdim,
      cutoff=cutoff,
      use_relative_cutoff=false,
      use_absolute_cutoff=true,
    )
  
    vals = diag(array(S))
    (length(vals) == 1 && vals[1]^2 ≤ cutoff) && return nothing,nothing,nothing
  
    return U,S,V
  end


##these are also not sane defaults if we want to keep the expansion at single site cost
_svd_solve_randomized_precontract(T,envMap, left_ind ; maxdim, cutoff,flux) = _svd_solve_randomized_precontract(envMap, left_ind, maxdim,maxdim;maxdim,cutoff)
###FIXME: override needs to be removed later, when rsvd for Map/Vector of ITensors is implemented
#_svd_solve_randomized_precontract(T,envMap, left_ind;flux, maxdim, cutoff) = _svd_solve_randomized_precontract(T,envMap, left_ind ; maxdim, cutoff) 
_svd_solve_randomized_precontract(T,envMap,envMapDag, left_ind;flux, maxdim, cutoff) = _svd_solve_randomized_precontract(T,envMap, left_ind ; maxdim, cutoff,flux)


###to do tomorrow:
#a) work on figuring out reasonable defaults for n,p ---> implement a function, + possibility to pass an aux function to the svd backend (probably via namedtuple)
#b) implement randomized SVD for ITensorNetworkMap or Vector of ITensors
##b1) figure out ordering of ITensorNetworkMap again
##b2) maybe streamline the logic everywhere so we don't have to dig back in every time we touch the code
##b3) ...