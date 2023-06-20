using Revise
using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Observers
using HDF5

include("observer.jl")

## rung decoupled heisenberg chain model ##
function rung_decoupl_heisenberg(N; J1=1.0, J2=0.5)
  H = OpSum()

  for e1 in 1:N-2
    e2 = e1 + 2

    J = iseven(e1) ? J1 : J2

    H += J / 2, "S+", e1, "S-", e2
    H += J / 2, "S-", e1, "S+", e2
    H += J, "Sz", e1, "Sz", e2
  end
  return H
end

let
  tmax = 2.
  dt = 0.1
  N = 10

  # g = named_binary_tree(3)
  # g = named_grid((12,1))

  # with QNs
  s = siteinds("S=1", N; conserve_qns=true)
  ψ = random_mps(s; states=(x-> isodd(x) ? "Up" : "Dn"), internal_inds_space=4)
  #ψ = random_mps(s; states=(x->["Up","Up","Up","Dn","Up","Dn","Up","Dn","Up","Up"][x]), internal_inds_space=1)
  model = heisenberg(chain_lattice_graph(N); J1= 1., h = 0.)
  H = mpo(model, s)

  # without Qns
  # g = chain_lattice_graph(N)
  # s = siteinds("S=1/2", g)
  # ψ = TTN(ITensorNetwork(s, "Up"))
  # model = ising(g; J1=-1., h=-1.) 
  # # model = rung_decoupl_heisenberg(N)
  # H = TTN(model, s)

  obs1 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) # "xPol" => return_x, 
  obs2 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) # "xPol" => return_x, 
  obs3 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) # "xPol" => return_x, 

  ################### DMRG #################

  # dmrg_x_kwargs = (nsweeps=10, reverse_step=false, normalize=true, maxdim=12, cutoff=1e-10, outputlevel=1, nsites = 1,)
  #
  # println("============== 1-site DMRG with subspace expansion ====================")
  # ϕ1 = dmrg_x(H, ψ; dmrg_x_kwargs..., expand = true)
  # # println("============== 2-site with subspace expansion ====================")
  # # ϕ2 = dmrg_x(H, ψ; dmrg_x_kwargs..., nsites = 2)
  # println("============== 1-site DMRG without subspace expansion ====================")
  # ϕ3 = dmrg_x(H, ψ; dmrg_x_kwargs..., expand = false)

  ################### TDVP #################

  D = 50
  tdvp_kwargs = (time_step = -im*dt, reverse_step=true, normalize=true, maxdim=D, cutoff=1e-4, outputlevel=1, nsite=1,)
  expander_cache=Any[]

  println("============== 1-site TDVP with 2-site subspace expansion ====================")

  # ϕ1 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expand="2-site", 
  #           (observer!)=obs1,
  #      )
  # @time begin
  # ϕ1 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           # expand="2-site", 
  #           expander_backend="full", 
  #           svd_backend="svd", 
  #           (observer!)=obs1,
  #      )

  # println("============== 1-site TDVP with full subspace expansion ====================")

  ϕ2 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
            # expand="full", 
            expander_backend="two_site", 
            svd_backend="krylov", 
            expander_cache=expander_cache,
            (observer!)=obs2,
       )

  # println("============== 1-site TDVP with krylov subspace expansion ====================")

  # @time begin
  # ϕ3 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expand="full", 
  #           (observer!)=obs3,
  #          )
  # end

  # name_obs1 = "data/rungDecoupHeis_krylov_exp_N=$(N)_D=$(D)"
  #
  # name_obs1 = "data/rungDecoupHeis_twosite_exp_N=$(N)_D=$(D)"
  # name_obs2 = "data/rungDecoupHeis_full_exp_N=$(N)_D=$(D)"
  # savedata(name_obs1, obs1)
  # savedata(name_obs2, obs2)
  # savedata(name_obs3, obs3)
end
nothing
