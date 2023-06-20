
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

# Define observer
function step(; sweep, bond, half_sweep)
  if bond == 1 && half_sweep ==2
    return sweep
  end
  return nothing
end

function current_time(; current_time, bond, half_sweep)
  if bond == 1 && half_sweep ==2
    return abs(current_time)
  end
  return nothing
end

function return_x(; psi, bond, half_sweep)
  if bond == 1 && half_sweep ==2
    expec = expect("Sx", psi)
    return real(sum(expec)/length(expec))
  end
  return nothing
end

function return_z(; psi, bond, half_sweep)
  if bond == 1 && half_sweep ==2
    expec = expect("Sz", psi)
    return real(sum(expec)/length(expec))
  end
  return nothing
end

function return_en(; psi, PH, bond, half_sweep)
  if bond == 1 && half_sweep ==2
    H = PH.H
    return real(inner(psi', H, psi) / inner(psi, psi))  
  end
  return nothing
end

# function return_dw(; psi, bond, half_sweep)
# if bond == 1 && half_sweep ==2
#     xxcorr = correlation_matrix(psi, "X","X")
#     D = 0
#     for bond in nearest_neighboursy(L, snake_curve(L))
#         D += 0.5 * (1 - xxcorr[bond[1],bond[2]])
#     end
#     return real(D)
# end
# return nothing
# end

# function return_entropy(; psi, bond, half_sweep)
# if bond == 1 && half_sweep ==2
#     pos = Int(N/2)
#     psi_ = orthogonalize(psi, pos)
#     _,S,_ = svd(psi_[pos], (commonind(psi_[pos-1], psi_[pos]), siteind(psi_,pos)))
#     SvN = 0
#     for n=1:dim(S,1)
#         p = S[n,n]^2
#         SvN -= p*log(p)
#     end
#     return SvN
# end
# return nothing
# end

function return_state(; psi, bond, half_sweep)
if bond == 1 && half_sweep ==2
    return psi
end
return nothing
end
