function _kkit_init_check(u₀,theadj,thenormal)
    β₀ = norm(u₀)
    #@show β₀
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    v₀ = theadj(u₀)
    #@show norm(v₀)
    α = norm(v₀) / β₀
    Av₀ = thenormal(v₀) # apply operator
    #@show dot(u₀, Av₀)
    α² = dot(u₀, Av₀) / β₀^2
    if norm(α) < sqrt(eps(Float64))
      return false
    else
      α² ≈ α * α || throw(ArgumentError("operator and its adjoint are not compatible"))
    end
    return true
  end
  

function _build_USV_without_QN(vals, lvecs, rvecs)
    # attach trivial index to left/right eigenvectors to take directsum over it
    lvecs = map(lvecs) do lvec 
      dummy_ind = Index(1; tags="u") 
      return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
    end
  
    rvecs = map(rvecs) do rvec 
      dummy_ind = Index(1; tags="v")
      return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
    end
  
    ### perform directsum
    U,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="u", 
    ), lvecs[2:end]; init=lvecs[1])
  
    V,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="v",
    ), rvecs[2:end]; init=rvecs[1])
  
    S = ITensors.diagITensor(vals, filter(v->hastags(v, "u"), inds(U)), filter(v->hastags(v, "v"), inds(V)))
  
    return U,S,V
  end
  
  function _build_USV_with_QN(vals_col, lvecs_col, rvecs_col, remainingSpaces;envflux)
    # attach trivial index to left/right eigenvectors to take directsum over it
    lvecs_col = map(zip(remainingSpaces,lvecs_col)) do (s,lvecs)
      return map(lvecs) do lvec 
        dummy_ind = Index(Pair(first(s),1); tags="u", dir=ITensors.In)
        return ITensor(array(lvec), inds(lvec)..., dummy_ind) => dummy_ind
      end
    end
  
    rvecs_col = map(zip(remainingSpaces,rvecs_col)) do (s,rvecs)
      return map(rvecs) do rvec
        rvec=rvec 
        dummy_ind = Index(Pair(envflux-first(s),1); tags="v", dir=ITensors.In)
        return ITensor(array(rvec), inds(rvec)..., dummy_ind) => dummy_ind
      end
    end
  
    ### perform directsum of left/right eigenvectors for each sector
    lvecs_col = map(lvecs_col) do lvecs
      l, inds_l = reduce((x,y) -> ITensors.directsum(
        x, y; tags="u",
      ), lvecs[2:end]; init=lvecs[1])
      length(lvecs) == 1 && return l => inds_l
      C = combiner(inds_l; tags="u", dir=dir(inds_l))
      l = l*C
      return l => commonind(l,C)
    end
  
    rvecs_col = map(rvecs_col) do rvecs
      r, inds_r = reduce((x,y) -> ITensors.directsum(
        x, y; tags="v",
      ), rvecs[2:end]; init=rvecs[1])
      length(rvecs) == 1 && return r => inds_r
      C = combiner(inds_r; tags="v", dir=dir(inds_r))
      r = r*C
      return r => commonind(r,C)
    end
  
    ### perform directsum over all sectors
    U,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="u" 
    ), lvecs_col[2:end]; init=lvecs_col[1])
  
    V,_ = reduce((x,y) -> ITensors.directsum(
      x, y; tags="v",
    ), rvecs_col[2:end]; init=rvecs_col[1])
  
    S = ITensors.diagITensor(vcat(vals_col...), filter(v->hastags(v, "u"), dag(inds(U))), filter(v->hastags(v, "v"), dag(inds(V))))
    return U,S,V
  end
  
  function _truncate_blocks!(d, vals_col, lvecs_col, rvecs_col, remainingSpaces, cutoff, maxdim) 
    d .= d .^ 2
    sort!(d; rev=true)
    docut=0.0
  
    ## implements what truncate! is supposed to do
    if length(d)==1 && d[1]<cutoff
       docut=d[1]+eps(eltype(d))
    elseif length(d)>1
      pos=argmin(d .>= cutoff)
      docut=d[min(maxdim+1,pos)]
    end
    
    dropblocks = Int[]
  
    for (n,vals) in enumerate(vals_col)
      full_dim = length(vals)
      blockdim = 0
      val = vals[blockdim + 1]^2
      while blockdim + 1 ≤ full_dim && val > docut
        blockdim += 1
        (blockdim + 1 ≤ full_dim) && (val = vals[blockdim + 1]^2)
      end
  
      if blockdim == 0
        push!(dropblocks,n)
      else
        vals_col[n]  = vals[1:blockdim]
        lvecs_col[n] = lvecs_col[n][1:blockdim]
        rvecs_col[n] = rvecs_col[n][1:blockdim]
      end
    end
    d.=sqrt.(d)
    deleteat!(vals_col,  dropblocks)
    deleteat!(lvecs_col, dropblocks)
    deleteat!(rvecs_col, dropblocks)
    deleteat!(remainingSpaces,  dropblocks)
  end
  
  function _krylov_svd_solve(T,
    envMap,envMapDag, left_ind; flux=flux(contract(envMap)),maxdim, cutoff,# kwargs...
  )
    maxdim = min(maxdim, 15)
    krylov_args = (tol = cutoff, krylovdim = maxdim, maxiter = 1)
    if !hasqns(left_ind)
      trial = randomITensor(eltype(envMap), left_ind)
      trial = trial / norm(trial)
  
      @timeit_debug timer "krylov check" begin
        _kkit_init_check(trial,envMapDag,envMap) || (return nothing,nothing,nothing)
      end
  
      try
        
        @timeit_debug timer "krylov svd solve" begin
          vals, lvecs, rvecs, info = KrylovKit.svdsolve(
            (x -> envMap*x, y -> envMapDag*y), trial; krylov_args...,
          )
        end
        #println("krylov solve worked on KrylovKit side")
      catch e 
        @show e
        return _svd_solve_normal(envMap, left_ind; maxdim, cutoff)
      end
  
      ## cutoff unnecessary values and expand the resulting vectors by a dummy index
      @timeit_debug timer "build USV from Krylov vecs" begin
        vals = filter(v->v^2≥cutoff, vals)[1:min(maxdim,end)]
        (length(vals) == 0) && return nothing,nothing,nothing
        lvecs = lvecs[1:min(length(vals),end)]
        rvecs = rvecs[1:min(length(vals),end)]
        U,S,V = _build_USV_without_QN(vals, lvecs, rvecs)
      end
  
    else
      vals_col  = Any[]
      lvecs_col = Any[]
      rvecs_col = Any[]
      d = Vector{real(T)}()
      remainingSpaces = Any[]
  
      for s in space(left_ind[1])
        last(s)==0 && continue
  
        theqn=first(s)
        trial = randomITensor(T, theqn, left_ind)
        adjtrial=envMapDag*trial
        trial2=envMap*adjtrial
        
        if size(storage(adjtrial))==(0,) || size(storage(trial2))==(0,)
          continue
        elseif ! _kkit_init_check(trial, envMapDag,envMap)
          continue
        end
        adjtrial=nothing
        trial2=nothing
        try
          @timeit_debug timer "krylov svd solve" begin
            vals, lvecs, rvecs, info = KrylovKit.svdsolve(
              (x -> envMap*x, y -> envMapDag*y), trial; krylov_args...,
            )
          end
        catch e 
          @show e
          return _svd_solve_normal(envMap, left_ind; maxdim, cutoff)
        end
        push!(vals_col,  vals)
        push!(lvecs_col, lvecs)
        push!(rvecs_col, conj.(dag.(rvecs)))  ###is the conjugate here justified or not?  
        push!(remainingSpaces, s)
        append!(d, vals)
      end
      (length(d) == 0) && return nothing,nothing,nothing
  
      @timeit_debug timer "build USV from Krylov vecs" begin
        _truncate_blocks!(d, vals_col, lvecs_col, rvecs_col, remainingSpaces, cutoff, maxdim)
  
        (length(vals_col) == 0) && return nothing,nothing,nothing
        U,S,V = _build_USV_with_QN(vals_col, lvecs_col, rvecs_col, remainingSpaces; envflux=flux)
      end
    end
    #println("krylov solve worked till the end")
    return U,S,V
  end