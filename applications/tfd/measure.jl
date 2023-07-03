function ITensors.MPS(psi::TreeTensorNetwork)
    if !ITensorNetworks.is_path_graph(siteinds(psi))
        error("Trying to convert to MPS although TTN is not a path graph. Exiting.")
    end
    orts=ortho_center(psi)
    sort!(orts)
    ortrange=first(orts):last(orts)
  
    psimps=ITensors.MPS([psi[i] for i in 1:nv(psi)])
    ITensors.set_ortho_lims!(psimps,ortrange)
    return psimps
end
  
function measure_corr_ij(kets,bras,i,j; conjugate=false)
    if conjugate
      return inner(Base.conj(bras[i]),kets[j])  ###lookup inner whether inner automatically applies dag or not
    else
      return inner(bras[i],kets[j])
    end
  end
  
function measure_corr(ket,bra;conjugate=true)
    #println("in single corr")
    if conjugate
        return inner(Base.conj(bra),ket)  ###lookup inner whether inner automatically applies dag or not
    else
        return inner(bra,ket)
    end
    
end

function measure_corr(i::Int,sendpsi::ITensors.MPS,recvpsi::ITensors.MPS,sendpsi0::ITensors.MPS)
    root=i
    recvpsi=MPI.bcast(sendpsi,root,comm)
    gf = measure_corr(sendpsi,recvpsi;conjugate=true)
    gfhalf = measure_corr(sendpsi0,recvpsi;conjugate=false)
    allgf=MPI.Allgather(gf, comm)
    allgfhalf=MPI.Allgather(gfhalf, comm)
    #@show MPI.Comm_rank(comm), i, allgf, allgfhalf 
    return allgf,allgfhalf
end

function measure_corr(sendpsi::MPS,recvpsi::MPS,sendpsi0::MPS;stride=3)
    Nphys_sites=div(length(sendpsi),stride)
    results=Vector{Vector{ComplexF64}}()
    results_half=Vector{Vector{ComplexF64}}()
    for i in 0:Nphys_sites-1
        gfs,gfs_half= measure_corr(i,sendpsi,recvpsi,sendpsi0)
        push!(results,gfs)
        push!(results_half,gfs_half)
    end
    return results,results_half
end
  
function measure_corr(sendpsi::TTN,recvpsi::TTN,sendpsi0::TTN;stride=3)
    measure_corr(MPS(sendpsi),MPS(recvpsi),MPS(sendpsi0);stride=stride)
end

function measure_autocorr(sendpsi::TTN)
   #println("in autocorr")
    measure_corr(MPS(sendpsi),MPS(sendpsi);conjugate=true)
end