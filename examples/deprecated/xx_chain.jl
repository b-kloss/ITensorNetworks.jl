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

  for e1 in 1:(N - 1)
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
  tmax = 40.0
  dt = 0.1
  N = 100
  D = 40
  Dexpand = 2000

  # Heisenberg with QNs
  s = siteinds("S=1/2", N; conserve_qns=true)
  states = vcat(fill("Up", Int(N//2)), fill("Dn", Int(N//2)))
  ψ = random_mps(s; states=(x -> states[x]), internal_inds_space=1)
  model = xx_chain(N)
  H = mpo(model, s)

  model_name = "xx_chain"
  params = "_N=$(N)_D=$(D)"

  obs1 = Observer(
    "time" => step, "zPol" => return_z, "energy" => return_en, "entropy" => return_entropy
  )
  obs2 = Observer(
    "time" => step, "zPol" => return_z, "energy" => return_en, "entropy" => return_entropy
  )
  obs3 = Observer(
    "time" => step, "zPol" => return_z, "energy" => return_en, "entropy" => return_entropy
  )
  obs4 = Observer(
    "time" => step, "zPol" => return_z, "energy" => return_en, "entropy" => return_entropy
  )
  obs5 = Observer(
    "time" => step, "zPol" => return_z, "energy" => return_en, "entropy" => return_entropy
  )

  name_obs1 = "data/" * model_name * "_two_site_svd" * params
  name_obs2 = "data/" * model_name * "_two_site_krylov" * params
  name_obs3 = "data/" * model_name * "_full_svd" * params
  name_obs4 = "data/" * model_name * "_full_krylov" * params
  name_obs5 = "data/" * model_name * "_two_site_general" * params

  tdvp_kwargs = (
    time_step=-im * dt,
    reverse_step=false,
    normalize=true,
    maxdim=D,
    maxdim_expand=Dexpand,
    outputlevel=1,
    cutoff_compress=1e-12,
    cutoff=1e-12,
  )

  Random.seed!(121)
  println("================================================================")

  ϕ1 = @time tdvp(
    H,
    -im * tmax,
    ψ;
    tdvp_kwargs...,
    expander_backend="two_site",
    svd_backend="svd",
    nsite=1,
    (step_observer!)=obs1,
  )
  savedata(name_obs1, obs1)

  println("================================================================")

  ϕ2 = @time tdvp(
    H,
    -im * tmax,
    ψ;
    tdvp_kwargs...,
    expander_backend="two_site",
    svd_backend="krylov",
    nsite=1,
    (step_observer!)=obs2,
  )
  savedata(name_obs2, obs2)

  println("================================================================")

  expander_cache = Any[]
  ϕ3 = @time tdvp(
    H,
    -im * tmax,
    ψ;
    tdvp_kwargs...,
    expander_backend="full",
    svd_backend="svd",
    nsite=1,
    expander_cache=expander_cache,
    (step_observer!)=obs3,
  )
  savedata(name_obs3, obs3)

  println("================================================================")

  expander_cache = Any[]
  ϕ4 = @time tdvp(
    H,
    -im * tmax,
    ψ;
    tdvp_kwargs...,
    expander_backend="full",
    svd_backend="krylov",
    nsite=1,
    expander_cache=expander_cache,
    (step_observer!)=obs4,
  )
  savedata(name_obs4, obs4)

  println("================================================================")

  # ϕ5 = @time tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expander_backend="none", 
  #           svd_backend="krylov",
  #           nsite=2,
  #           (step_observer!)=obs5,
  #      )
  # savedata(name_obs5, obs5)

end
nothing
