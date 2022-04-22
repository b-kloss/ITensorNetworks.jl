import Base:
  # types
  Vector,
  # functions
  convert,
  copy,
  getindex,
  hvncat,
  setindex!,
  show,
  isassigned

import .DataGraphs: underlying_graph, vertex_data, edge_data

import Graphs: Graph

import NamedGraphs: vertex_to_parent_vertex

import ITensors:
  # contraction
  contract,
  # site and link indices
  siteind,
  siteinds,
  linkinds,
  # index set functions
  uniqueinds,
  commoninds,
  replaceinds,
  # priming and tagging
  adjoint,
  sim,
  prime,
  setprime,
  noprime,
  replaceprime,
  addtags,
  removetags,
  replacetags,
  settags,
  # dag
  dag

import ITensors.ContractionSequenceOptimization:
  optimal_contraction_sequence

using ITensors.ContractionSequenceOptimization:
  deepmap

import ITensors.ITensorVisualizationCore:
  visualize