# util function to save data in obs
function savedata(name::String, obs)
  name == "" && return
  h5open(name*".h5", "w") do file    
    # iterate through the fields of obs and append the data to the dataframe
    for n in names(obs)
      create_group(file, n)
      for (i,data) in enumerate(obs[:,n])
        file[n][string(i)] = [data[i] for i in 1:length(data)]
      end
    end 
  end
end

function MPS(psi::TTN)
    if !is_path_graph(siteinds(psi))
        error("Trying to convert to MPS although TTN is not a path graph. Exiting.")
    end
    orts=ortho_center(psi)
    sort!(orts)
    ortrange=first(orts):last(orts)
    
    psimps=ITensors.MPS([psi[i] for i in 1:nv(psi)])
    ITensors.set_ortho_lims!(psimps,ortrange)
    return psimps
end

# overload expect function to be more efficient for TTNs
function expect(
  op::String,
  ψ::ITensorNetworks.AbstractTreeTensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=false,
  sequence=nothing,
  vertices=vertices(ψ),
)
  ψC = copy(ψ)
  s = siteinds(ψ)
  ElT = ITensorNetworks.promote_itensor_eltype(ψ)
  res = Dictionary(vertices, Vector{ElT}(undef, length(vertices)))

  for v in vertices
    O = ITensor(Op(op, v), s)
    ψC = orthogonalize(ψC, v)
    ψCv = ψC[v]
    res[v] = inner(ψCv, apply(O, ψCv)) / inner(ψCv, ψCv)
  end
  return res
end

# Define stepobserver
function step(; sweep)
  return sweep
end

## just use step 
# function current_time(; current_time)
#   return abs(current_time)
# end

function return_x(; psi)
  expec = expect("Sx", psi)
  return real(sum(expec)/length(expec))
end

function return_z(; psi)
  expec = expect("Sz", psi)
  return real.(collect(expec))
end

function return_z_mps(; psi)
  expec = ITensors.expect(MPS(psi), "Sz")
  return real.(collect(expec))
end

function return_z_mean(; psi)
  expec = expect("Sz", psi)
  return real(sum(expec)/length(expec))
end

function return_z_half(; psi)
  half_len = floor(Int, length(vertices(psi))/2)
  expec = expect("Sz", psi; vertices=half_len)
  return real(sum(expec)/length(expec))
end

function return_en(; psi, PH)
  H = PH.H
  return real(inner(psi', H, psi) / inner(psi, psi))  
end

function return_state(; psi, end_of_sweep)
  if end_of_sweep
      return psi
  end
  return nothing
end

function return_entropy(; psi)
  pos = floor(Int, length(vertices(psi))/2)
  psi_ = orthogonalize(psi, pos)
  _,S,_ = svd(psi_[pos], (commonind(psi_[pos-1], psi_[pos]), filter(i->hastags(i, "Site"), inds(psi_[pos]))))
  SvN = 0
  for n=1:dim(S,1)
      p = S[n,n]^2
      SvN -= p*log(p)
  end
  return SvN
end

# # Define observer
# function step(; sweep, end_of_sweep)
#   if bond == 1 && half_sweep ==2
#     return sweep
#   end
#   return nothing
# end
#
# function current_time(; current_time, end_of_sweep)
#   if end_of_sweep
#     return abs(current_time)
#   end
#   return nothing
# end
#
# function return_x(; psi, end_of_sweep)
#   if end_of_sweep
#     expec = expect("Sx", psi)
#     return real(sum(expec)/length(expec))
#   end
#   return nothing
# end
#
# function return_z(; psi, end_of_sweep)
#   if end_of_sweep
#     expec = expect("Sz", psi)
#     return real.(collect(expec))
#   end
#   return nothing
# end
#
# function return_z_mps(; psi, end_of_sweep)
#   if end_of_sweep
#     expec = ITensors.expect(MPS(psi), "Sz")
#     return real.(collect(expec))
#   end
#   return nothing
# end
#
# function return_z_mean(; psi, end_of_sweep)
#   if end_of_sweep
#     expec = expect("Sz", psi)
#     return real(sum(expec)/length(expec))
#   end
#   return nothing
# end
#
# function return_z_half(; psi, end_of_sweep)
#   if end_of_sweep
#     half_len = floor(Int, length(vertices(psi))/2)
#     expec = expect("Sz", psi; vertices=half_len)
#     return real(sum(expec)/length(expec))
#   end
#   return nothing
#
# end
#
# function return_en(; psi, PH, end_of_sweep)
#   if end_of_sweep
#     H = PH.H
#     return real(inner(psi', H, psi) / inner(psi, psi))  
#   end
#   return nothing
# end
#
# function return_state(; psi, end_of_sweep)
#   if end_of_sweep
#       return psi
#   end
#   return nothing
# end
#
# function return_entropy(; psi, end_of_sweep)
#     if end_of_sweep
#     pos = floor(Int, length(vertices(psi))/2)
#     psi_ = orthogonalize(psi, pos)
#     _,S,_ = svd(psi_[pos], (commonind(psi_[pos-1], psi_[pos]), filter(i->hastags(i, "Site"), inds(psi_[pos]))))
#     SvN = 0
#     for n=1:dim(S,1)
#         p = S[n,n]^2
#         SvN -= p*log(p)
#     end
#     return SvN
# end
# return nothing
# end
