# util function to save data in obs
function savedata(name::String, obs)
  name == "" && return nothing
  h5open(name * ".h5", "w") do file
    # iterate through the fields of obs and append the data to the dataframe
    for n in names(obs)
      create_group(file, n)
      for (i, data) in enumerate(obs[:, n])
        file[n][string(i)] = [data[i] for i in 1:length(data)]
      end
    end
  end
end

function MPS(state::TTN)
  if !is_path_graph(siteinds(state))
    error("Trying to convert to MPS although TTN is not a path graph. Exiting.")
  end
  orts = ortho_center(state)
  sort!(orts)
  ortrange = first(orts):last(orts)

  statemps = ITensors.MPS([state[i] for i in 1:nv(state)])
  ITensors.set_ortho_lims!(statemps, ortrange)
  return statemps
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
function step(; which_sweep)
  return which_sweep
end

## just use step 
# function current_time(; current_time)
#   return abs(current_time)
# end

function return_x(; state)
  expec = expect("Sx", state)
  return real(sum(expec) / length(expec))
end

function return_z(; state)
  expec = expect("Sz", state)
  return real.(collect(expec))
end

function return_z_mps(; state)
  expec = ITensors.expect(MPS(state), "Sz")
  return real.(collect(expec))
end

function return_z_mean(; state)
  expec = expect("Sz", state)
  return real(sum(expec) / length(expec))
end

function return_z_half(; state)
  half_len = floor(Int, length(vertices(state)) / 2)
  expec = expect("Sz", state; vertices=half_len)
  return real(sum(expec) / length(expec))
end

function return_en(; state, PH)
  H = PH.H
  return real(inner(state', H, state) / inner(state, state))
end

function return_state(; state, end_of_sweep)
  if end_of_sweep
    return state
  end
  return nothing
end

function return_entropy(; state)
  pos = floor(Int, length(vertices(state)) / 2)
  state_ = orthogonalize(state, pos)
  _, S, _ = svd(
    state_[pos],
    (
      commonind(state_[pos - 1], state_[pos]),
      filter(i -> hastags(i, "Site"), inds(state_[pos])),
    ),
  )
  SvN = 0
  for n in 1:dim(S, 1)
    p = S[n, n]^2
    SvN -= p * log(p)
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
# function return_x(; state, end_of_sweep)
#   if end_of_sweep
#     expec = expect("Sx", state)
#     return real(sum(expec)/length(expec))
#   end
#   return nothing
# end
#
# function return_z(; state, end_of_sweep)
#   if end_of_sweep
#     expec = expect("Sz", state)
#     return real.(collect(expec))
#   end
#   return nothing
# end
#
# function return_z_mps(; state, end_of_sweep)
#   if end_of_sweep
#     expec = ITensors.expect(MPS(state), "Sz")
#     return real.(collect(expec))
#   end
#   return nothing
# end
#
# function return_z_mean(; state, end_of_sweep)
#   if end_of_sweep
#     expec = expect("Sz", state)
#     return real(sum(expec)/length(expec))
#   end
#   return nothing
# end
#
# function return_z_half(; state, end_of_sweep)
#   if end_of_sweep
#     half_len = floor(Int, length(vertices(state))/2)
#     expec = expect("Sz", state; vertices=half_len)
#     return real(sum(expec)/length(expec))
#   end
#   return nothing
#
# end
#
# function return_en(; state, PH, end_of_sweep)
#   if end_of_sweep
#     H = PH.H
#     return real(inner(state', H, state) / inner(state, state))  
#   end
#   return nothing
# end
#
# function return_state(; state, end_of_sweep)
#   if end_of_sweep
#       return state
#   end
#   return nothing
# end
#
# function return_entropy(; state, end_of_sweep)
#     if end_of_sweep
#     pos = floor(Int, length(vertices(state))/2)
#     state_ = orthogonalize(state, pos)
#     _,S,_ = svd(state_[pos], (commonind(state_[pos-1], state_[pos]), filter(i->hastags(i, "Site"), inds(state_[pos]))))
#     SvN = 0
#     for n=1:dim(S,1)
#         p = S[n,n]^2
#         SvN -= p*log(p)
#     end
#     return SvN
# end
# return nothing
# end
