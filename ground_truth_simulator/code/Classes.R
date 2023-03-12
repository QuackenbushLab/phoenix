#----define classes----
setClass(
  Class = 'Node',
  slots = list(
    name = 'character',
    spmax = 'numeric',
    spdeg = 'numeric',
    inedges = 'character',
    outedges = 'character',
    logiceqn = 'character'
  )
)

setClass(
  Class = 'NodeRNA',
  contains = 'Node',
  slots = list(
    tau = 'numeric'
  )
)

setClass(
  Class = 'Edge',
  slots = list(
    from = 'character',
    to = 'character',
    weight = 'numeric',
    name = 'character'
  )
)

setClass(
  Class = 'EdgeReg',
  contains = 'Edge',
  slots = list(
    EC50 = 'numeric',
    n = 'numeric',
    activation = 'logical'
  )
)

setClass(
  Class = 'GraphGRN',
  slots = list(
    nodeset = 'list',
    edgeset = 'list'
  )
)

setClass(
  Class = 'SimulationGRN',
  slots = list(
    graph = 'GraphGRN',
    expnoise = 'numeric',
    bionoise = 'numeric',
    seed = 'numeric',
    inputModels = 'list'
  )
)