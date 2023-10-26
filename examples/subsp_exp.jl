using Base: ForwardOrdering
using Revise
using NamedGraphs
using ITensors
using ITensorNetworks
#using ITensorUnicodePlots
using Observers
using ITensors.HDF5
using Random
using Dictionaries

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

## rung decoupled heisenberg chain model ##
function isinglin(N; J=-1., g=-1.)
  H = OpSum()
  for e1 in 1:N-1
    e2 = e1 + 1

    H += J , "Z", e1, "Z", e2
  end
  for e in 1:N
    H += g , "X", e
  end
  return H
end

  ## heisenberg chain model ##
function heisenberglin(N; J=1.)
  H = OpSum()
  for e1 in 1:(N-1)
    e2 = e1 + 1

    H += J / 2, "S+", e1, "S-", e2
    H += J / 2, "S-", e1, "S+", e2
    H += J, "Sz", e1, "Sz", e2
  end
  return H
end

let
  Random.seed!(1234)
  tmax = 2.
  dt = 0.025
  N = 16
  J = -1.
  g = -1.
  tau = 0.1
  D = 12

  # g = named_binary_tree(3)
  # g = named_grid((12,1))

  # with QNs
   s = siteinds("S=1/2", N; conserve_qns=true)
   ψ = random_mps(s; states=(x-> isodd(x) ? "Up" : "Dn"), internal_inds_space=1)
  # ψ = random_mps(s; states=(x->["Up","Up","Up","Dn","Up","Dn","Up","Dn","Up","Up"][x]), internal_inds_space=1)
   model = heisenberglin(N)
   H = mpo(model, s)
  @show s
  # without Qns
  #g = chain_lattice_graph(N)
  #s = siteinds("S=1/2", g)
  # ψ = TTN(ITensorNetwork(s, x->isodd(x) ? "Up" : "Dn"))
  #ψ = TTN(ITensorNetwork(s, "Up"))
  #model = isinglin(N; J=-1., g=-1.) 
  #H = TTN(model, s)

  # rung-decoupled Heisenberg without Qns
  # g = chain_lattice_graph(N)
  # s = siteinds("S=1", N)
  # ψ = random_mps(s; states= x->rand(["Up","Dn"]), internal_inds_space=4)
  # model = rung_decoupl_heisenberg(N)
  # H = mpo(model, s)
  #
  model_name = "ising"

  obs1 = Observer("zPol" => return_z) #, "en" => return_en)
  obs2 = Observer("zPol" => return_z) #, "en" => return_en)
  obs3 = Observer("zPol" => return_z) #, "en" => return_en) 
  obs4 = Observer("zPol" => return_z) #, "en" => return_en) 
  obs5 = Observer("zPol" => return_z) #, "en" => return_en) 
  name_obs1 = "data/"*model_name*"_two_site_svd_N=$(N)_D=$(D)"
  name_obs2 = "data/"*model_name*"_two_site_krylov_N=$(N)_D=$(D)"
  name_obs3 = "data/"*model_name*"_full_svd_N=$(N)_D=$(D)"
  name_obs4 = "data/"*model_name*"_full_krylov_N=$(N)_D=$(D)"
  name_obs5 = "data/"*model_name*"_two_site_general_N=$(N)_D=$(D)"

  ################### DMRG #################

  # dmrg_x_kwargs = (nsweeps=10, reverse_step=false, normalize=true, maxdim=12, cutoff=1e-10, outputlevel=1, nsites = 1,)
  # println("============== 1-site DMRG with subspace expansion ====================")
  # ϕ1 = dmrg_x(H, ψ; dmrg_x_kwargs..., expand = true)
  # # println("============== 2-site with subspace expansion ====================")
  # # ϕ2 = dmrg_x(H, ψ; dmrg_x_kwargs..., nsites = 2)
  # println("============== 1-site DMRG without subspace expansion ====================")
  # ϕ3 = dmrg_x(H, ψ; dmrg_x_kwargs..., expand = false)

  ################### TDVP #################
  tdvp_cutoff=1e-14
  tdvp_kwargs = (time_step = -im*dt, reverse_step=true, normalize=true, maxdim=D, cutoff=tdvp_cutoff,cutoff_expand=(tdvp_cutoff/dt), outputlevel=1)
  @show tdvp_kwargs
  println("================================================================")

  # ϕ1 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
  #           expander_backend="two_site", 
  #           svd_backend="svd",
  #           nsite=1,
  #           # expander_cache=expander_cache,
  #           (observer!)=obs1,
  #      )
  # savedata(name_obs1, obs1)
  #
  # println("================================================================")
  #
  # Random.seed!(1234)
  #
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

  expander_cache=Any[]
  ϕ5 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
            expander=ITensorNetworks._two_site_expand_core, 
            #step_expander=ITensorNetworks._full_expand_core_vertex, 
            
            #maxdim_expand=60,
            svd_func=ITensorNetworks._svd_solve_normal,
            nsite=1,
            expander_cache=expander_cache,
            (observer!)=obs5,
       )
 # @show obs5
  #savedata(name_obs5, obs5)

end
nothing
# using Revise
# using NamedGraphs
# using ITensors
# using ITensorNetworks
# using ITensorUnicodePlots
# using Observers
# using HDF5
# using Random
# include("observer.jl.bk")
#
# ## rung decoupled heisenberg chain model ##
# function rung_decoupl_heisenberg(N; J1=1.0, J2=0.5)
#   H = OpSum()
#
#   for e1 in 1:N-2
#     e2 = e1 + 2
#
#     J = iseven(e1) ? J1 : J2
#
#     H += J / 2, "S+", e1, "S-", e2
#     H += J / 2, "S-", e1, "S+", e2
#     H += J, "Sz", e1, "Sz", e2
#   end
#   return H
# end
#
# let
#   Random.seed!(1234)
#   tmax = 2
#   dt = 0.1
#   N = 10
#
#   # g = named_binary_tree(3)
#   # g = named_grid((12,1))
#
#   # with QNs
#   s = siteinds("S=1", N; conserve_qns=true)
#   ψ = random_mps(s; states=(x-> isodd(x) ? "Up" : "Dn"), internal_inds_space=1)
#   # ψ = random_mps(s; states=(x->["Up","Up","Up","Dn","Up","Dn","Up","Dn","Up","Up"][x]), internal_inds_space=1)
#   model = heisenberg(chain_lattice_graph(N); J1= 1.,h = 0.)
#   H = mpo(model, s)
#
#   # without Qns
#   # g = chain_lattice_graph(N)
#   # s = siteinds("S=1/2", N)
#   # # ψ = TTN(ITensorNetwork(s, "Up"))
#   # ψ = random_mps(s; states=(x-> isodd(x) ? "Up" : "Dn"), internal_inds_space=1)
#   # model = ising(chain_lattice_graph(N); J1=1., J2=-0., h=-1.) 
#   # # model = rung_decoupl_heisenberg(N)
#   # H = mpo(model, s)
#
#   obs1 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) # "xPol" => return_x, 
#   obs2 = Observer("time" => current_time, "zPol" => return_z_half) #, "en" => return_en) # "xPol" => return_x, 
#   obs3 = Observer("time" => current_time, "zPol" => return_z) #, "en" => return_en) # "xPol" => return_x, 
#
#   ################### DMRG #################
#
#   # dmrg_x_kwargs = (nsweeps=10, reverse_step=false, normalize=true, maxdim=12, cutoff=1e-10, outputlevel=1, nsites = 1,)
#   #
#   # println("============== 1-site DMRG with subspace expansion ====================")
#   # ϕ1 = dmrg_x(H, ψ; dmrg_x_kwargs..., expand = true)
#   # # println("============== 2-site with subspace expansion ====================")
#   # # ϕ2 = dmrg_x(H, ψ; dmrg_x_kwargs..., nsites = 2)
#   # println("============== 1-site DMRG without subspace expansion ====================")
#   # ϕ3 = dmrg_x(H, ψ; dmrg_x_kwargs..., expand = false)
#
#   ################### TDVP #################
#
#   D = 50
#   # tdvp_kwargs = (time_step = -im*dt, reverse_step=true, normalize=true, maxdim=D, cutoff=1e-10, outputlevel=1, nsite=1,)
#   expander_cache=Any[]
#
#   println("============== 1-site TDVP with 2-site subspace expansion ====================")
#   tdvp_kwargs = (time_step = -im*dt, reverse_step=true, normalize=true, maxdim=D, cutoff=1e-5, outputlevel=1, nsite=1,solver="none")
#   
#    ϕ1 = tdvp(H, -im*dt, ψ; tdvp_kwargs..., 
#              expander_backend="two_site",
#              svd_backend="krylov",
#              (observer!)=obs1,
#        )
#   # @time begin
#   # ϕ1 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
#   #           # expand="2-site", 
#   #           expander_backend="full", 
#   #           svd_backend="svd", 
#   #           (observer!)=obs1,
#   #      )
#
#   # println("============== 1-site TDVP with full subspace expansion ====================")
#   # @show tdvp_kwargs
#   # ϕ2 = tdvp(H, -im*tmax, ϕ1; tdvp_kwargs..., 
#   #           # expand="full", 
#   #           expander_backend="two_site", 
#   #           svd_backend="svd", 
#   #           expander_cache=expander_cache,
#   #           (observer!)=obs2,
#   #      )
#   #
#   # # println("============== 1-site TDVP with krylov subspace expansion ====================")
#   # @show obs2
#   # @time begin
#   # ϕ3 = tdvp(H, -im*tmax, ψ; tdvp_kwargs..., 
#   #           expand="full", 
#   #           (observer!)=obs3,
#   #          )
#   # end
#
#   # name_obs1 = "data/rungDecoupHeis_krylov_exp_N=$(N)_D=$(D)"
#   #
#   # name_obs1 = "data/rungDecoupHeis_twosite_exp_N=$(N)_D=$(D)"
#   # name_obs2 = "data/rungDecoupHeis_full_exp_N=$(N)_D=$(D)"
#   # savedata(name_obs1, obs1)
#   # savedata(name_obs2, obs2)
#   # savedata(name_obs3, obs3)
# end
# nothing
