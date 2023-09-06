function get_qn_dict(ind,theflux;auxdir=ITensors.In)
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
            # determine size of projection
            en = min(ndict[translate_qns[qn]],thedim)
            #ToDo: make this logic more transparent
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

function approx_the_same(o,n;abs_eps=1e-11,rel_eps=1e-4)
    absdev=abs.(o .- n)
    reldev=abs.((o .- n) ./ n)
    abs_conv = absdev .< abs_eps
    rel_conv = reldev .< rel_eps
    return all(abs_conv .|| rel_conv)
end

function is_converged_block(o,n;svd_kwargs...)
    ##returns the number of converged singular values above cutoff, and minimum among all converged singular values
    #@show o,n,typeof(o)
    #@show svd_kwargs
    maxdim=get(svd_kwargs, :maxdim,Inf)
    #@show maxdim
    ##if there is a cutoff arg, the svd will not return singular values below cutoff anyway...
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
    maxdim=get(svd_kwargs, :maxdim,Inf)
    os=sort(storage(oS),rev=true)
    ns=sort(storage(nS),rev=true)
    if length(os)>=maxdim && length(ns) >= maxdim
        conv_global=approx_the_same(os[1:maxdim],ns[1:maxdim])
    else
        r=min(length(os),length(ns))
        conv_global=approx_the_same(os[1:r],ns[1:r])
    end

    conv_bool_total=true
    
    soS=space(inds(oS)[1])
    snS=space(inds(nS)[1])
    qns=union(first.(soS),first.(snS))

    
    oblocks=eachnzblock(oS)
    oblockdict=Int.(getindex.(oblocks,1))
    oqnindtoblock=Dict(collect(values(oblockdict)) .=> collect(keys(oblockdict)))

    nblocks=eachnzblock(nS)
    nblockdict=Int.(getindex.(nblocks,1))
    nqnindtoblock=Dict(collect(values(nblockdict)) .=> collect(keys(nblockdict)))       
    
    for qn in qns
        if qn in first.(snS) && qn in first.(soS)
            oqnind=findall((first.(soS)) .== [qn,])[]
            nqnind=findall((first.(snS)) .== (qn,))[]
            oblock=oqnindtoblock[oqnind]
            nblock=nqnindtoblock[nqnind]
            
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
            ndict[qn]+=n_inc 
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

function rsvd_core(AC,M;svd_kwargs...)
    Q=AC*M
    Q=ITensors.qr(Q,commoninds(AC,Q))[1]
    QAC=dag(Q)*AC
    #@show svd_kwargs
    fact=svd(QAC,uniqueind(QAC,AC);svd_kwargs...)
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