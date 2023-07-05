using Revise
using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Observers
using HDF5
using Dictionaries
using Random

include("observer.jl")

## Heisenberg chain ##
function xx_chain(N; J=1.0)
  H = OpSum()

  for e1 in 1:N-1
    e2 = e1 + 1
    e2 = e2 > N ? 1 : e2
    H += J / 2, "S+", e1, "S-", e2
    H += J / 2, "S-", e1, "S+", e2
    # H += J / 2, "Sx", e1, "Sx", e2
    # H += J / 2, "Sy", e1, "Sy", e2
  end
  return H
end

let
  tmax = 20.
  dt = 0.1
  N = 100
  D = 40

  # Heisenberg with QNs
  s = siteinds("S=1/2", N; conserve_qns=true)
  states = vcat(fill("Up", Int(N//2)), fill("Dn", Int(N//2)))
  ψ = random_mps(s; states=(x->states[x]), internal_inds_space=1)
  model = xx_chain(N)
  H = mpo(model, s)

  model_name = "xx_chain"

  obs1 = Observer("time" => current_time, "zPolMPS" => return_z_mps, "zPol" => return_z) #, "energy" => return_en, "entropy" => return_entropy)
  obs2 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en)
  obs3 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) 
  obs4 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) 
  obs5 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) 
  name_obs1 = "data/"*model_name*"_two_site_svd_N=$(N)_D=$(D)"
  name_obs2 = "data/"*model_name*"_two_site_krylov_N=$(N)_D=$(D)"
  name_obs3 = "data/"*model_name*"_full_svd_N=$(N)_D=$(D)"
  name_obs4 = "data/"*model_name*"_full_krylov_N=$(N)_D=$(D)"
  name_obs5 = "data/"*model_name*"_two_site_general_N=$(N)_D=$(D)"

  tdvp_kwargs = (time_step = -im*dt, reverse_step=true, normalize=true, maxdim=D, cutoff=1e-12, outputlevel=1, cutoff_compress=1e-12)

  Random.seed!(121)
  println("================================================================")

  # expander_cache=Any[]
  ϕ1 = tdvp(H, -im*6.5, ψ; tdvp_kwargs...,  
            expander_backend="two_site", 
            svd_backend="svd",
            nsite=1,
            (observer!)=obs1,
       )
  ϕ1 = tdvp(H, -im*13.5, ϕ1; tdvp_kwargs...,  
            expander_backend="none", 
            svd_backend="svd",
            nsite=1,
            (observer!)=obs1,
       )
  savedata(name_obs1, obs1)

  println("================================================================")

  # ϕ2 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expander_backend="two_site", 
  #           svd_backend="krylov",
  #           nsite=1,
  #           # expander_cache=expander_cache,
  #           (observer!)=obs2,
  #      )
  # savedata(name_obs2, obs2)

  println("================================================================")

  # expander_cache=Any[]
  # ϕ3 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expander_backend="full", 
  #           svd_backend="svd",
  #           nsite=1,
  #           expander_cache=expander_cache,
  #           (observer!)=obs3,
  #      )
  # savedata(name_obs3, obs3)

  println("================================================================")

  # expander_cache=Any[]
  # ϕ4 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expander_backend="full", 
  #           svd_backend="krylov",
  #           nsite=1,
  #           expander_cache=expander_cache,
  #           (observer!)=obs4,
  #      )
  # savedata(name_obs4, obs4)

  println("================================================================")

  # expander_cache=Any[]
  # ϕ5 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expander_backend="none", 
  #           svd_backend="krylov",
  #           nsite=2,
  #           # expander_cache=expander_cache,
  #           (observer!)=obs5,
  #      )
  # savedata(name_obs5, obs5)

end
nothing
