function build_guess_matrix(eltype::Type{<:Number},ind,n::Int,p::Int;theflux=nothing)
    if hasqns(ind)
        thedir=dir(ind)
        auxdir=ITensors.In
        aux_spaces=Pair{QN,Int64}[]
        for s in space(ind)
            thedim = last(s)
            qn = first(s)
            # determine size of projection
            en = min(n+p,thedim)
            #ep = max(0,en-n)
            #ToDo: make this logic more transparent
            aux_qn = (-auxdir) * ((-thedir)*qn + theflux )
            push!(aux_spaces,Pair(aux_qn,en))
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

function build_guess_matrix(eltype::Type{<:Number},ind,ndict::Dict;theflux=nothing)
    if hasqns(ind)
        thedir=dir(ind)
        auxdir=ITensors.In
        aux_spaces=Pair{QN,Int64}[]
        for s in space(ind)
            thedim = last(s)
            qn = first(s)
            # determine size of projection
            en = min(ndict[s],thedim)
            #ToDo: make this logic more transparent
            aux_qn = (-auxdir) * ((-thedir)*qn + theflux )
            push!(aux_spaces,Pair(aux_qn,en))
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

function init_guess_sizes(cind,n::Int,rule)
    if hasqns(ind)
        ndict=Dict{Pair{QN, Int64},Int64}
        for s in space(ind)
            thedim = last(s)
            qn = first(s)
            ndict[s]=min(rule(n),thedim)            
        end
    else
        thedim=dim(ind)
        ndict=Dict{Int64,Int64}
        ndict[thedim]=min(thedim,rule(n))
    end
    return ndict
end

function increment_guess_sizes(ndict::Dict{Pair{QN, Int64},Int64},n_inc::Int,rule)
    for key in keys(ndict)
        thedim=last(key)
        ndict[key]=min(thedim,ndict[key]+rule(n_inc))
    end
    return ndict
end

function increment_guess_sizes(ndict::Dict{Int64,Int64},n_inc::Int,rule)
    for key in keys(ndict)
        thedim=key
        ndict[key]=min(thedim,ndict[key]+rule(n_inc))
    end
    return ndict
end

function increment_guess_sizes(ndict::Dict,old_fact,new_fact,n_inc;svd_kwargs...)
    oS=old_fact.S
    nS=new_fact.s
    for block in eachnzblock(oS)
        i=Int(block[1])
        #assumes same ordering of inds
        oldspace=space(inds(oS)[1])
        newspace=space(inds(nS)[1])
        #make sure blocks are the same QN when we compare them
        @assert first(oldspace[i])==first(newspace[i])'
        ovals=diag(oS[block])
        nvals=diag(nS[block])
        conv=is_converged_block(ovals,nvals;svd_kwargs)
    end
end

function rsvd_iterative(A::ITensor,linds::Vector{<:Index};svd_kwargs...)
    ##preprocess
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
    ##build ndict for initial run
    n_init=1
    p_rule(n)=2*n
    ndict=init_guess_sizes(cR,n_init,p_rule)
    M=build_guess_matrix(eltype(AC),cR,ndict;theflux=hasqns(AC) ? flux(AC) : nothing)
    fact=rsvd_core(AC,M;svd_kwargs)
    n_inc=1
    ndict=increment_guess_sizes(ndict,n_inc,p_rule)
    while true
        M=build_guess_matrix(eltype(AC),cR,ndict;theflux=hasqns(AC) ? flux(AC) : nothing)
        new_fact=rsvd_core(AC,M;svd_kwargs...)
        if is_converged(fact,new_fact;svd_kwargs...)
            break
        else
            
        #compare convergence
        end
    end

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
    fact=rsvd_core(AC,M;svd_kwargs)
    return dag(CL)*Q*fact.U, fact.S, fact.V*dag(CR)
    #return ITensors.TruncSVD(dag(CL)*Q*fact.U,fact.S,fact.V*dag(CR),fact.spec,linds,rinds)  #?
end

function rsvd_core(AC,M;svd_kwargs)
    Q=AC*M
    Q=ITensors.qr(Q,commoninds(AC,Q))[1]
    QAC=dag(Q)*AC
    fact=svd(QAC,uniqueind(QAC,AC);svd_kwargs...)
    return fact
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