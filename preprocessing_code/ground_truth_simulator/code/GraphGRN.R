#----Node----
setValidity('Node', validNode)

setMethod(
	f = 'initialize',
	signature = 'Node',
	definition = initNode
)

setMethod(
	f = '$',
	signature = 'Node',
	definition = function(x, name) {
		return(slot(x, name))
	}
)

setMethod(
	f = '$<-',
	signature = 'Node',
	definition = function(x, name, value) {
		slot(x, name)<-value
		validObject(x)
		return(x)
	}
)

setMethod(
  f = 'show',
  signature = 'Node',
  definition = function(object) {
    mxp = 10
    
    cat(paste0('***', class(object), '***'), '\n')
    cat('Name:', object@name, '\n')
    cat('RNA max:', object@spmax, '\n')
    cat('RNA degradation rate:', object@spdeg, '\n')
    cat('Logic equation:', object@logiceqn, '\n')
    
    #print edge list
    cat('Incoming edges:', maxPrint(object@inedges, mxp), '\n')
    cat('Outgoing edges:', maxPrint(object@outedges, mxp), '\n')
  }
)

#----NodeRNA----
setValidity('NodeRNA', validNodeRNA)

setMethod(
	f = 'initialize',
	signature = 'NodeRNA',
	definition = initNodeRNA
)

setMethod(
	f = 'show',
	signature = 'NodeRNA',
	definition = function(object) {
	  callNextMethod()
		cat('Time const:', object@tau, '\n')
	}
)

setGeneric(
	name = 'generateEqn',
	def = function(node, graph){
		standardGeneric('generateEqn')
	}
)

setMethod(
	f = 'generateEqn',
	signature = c('NodeRNA', 'GraphGRN'),
	definition = generateRateEqn
)

#----Edge----
setValidity('Edge', validEdge)

setMethod(
	f = '$',
	signature = 'Edge',
	definition = function(x, name) {
		return(slot(x, name))
	}
)

setMethod(
	f = '$<-',
	signature = 'Edge',
	definition = function(x, name, value) {
		if (name %in% 'name') {
			stop('Edge name is generated automatically. It cannot be modified.')
		}
		
		slot(x, name)<-value
		validObject(x)
		return(x)
	}
)

setMethod(
	f = 'show',
	signature = 'Edge',
	definition = function(object) {
		mxp = 10
		
		cat(paste0('***', class(object), '***'), '\n')
		cat('Name:', object@name, '\n')
		cat('From:', maxPrint(object@from, mxp), '\n')
		cat('To:', object@to, '\n')
		cat('Weight:', object@weight, '\n')
	}
)

setGeneric(
	name = 'generateActivationEqn',
	def = function(object){
		standardGeneric('generateActivationEqn')
	}
)

#----EdgeReg----
setValidity('EdgeReg', validEdgeReg)

setMethod(
  f = 'initialize',
  signature = 'EdgeReg',
  definition = initEdgeReg
)

setMethod(
	f = 'generateActivationEqn',
	signature = 'EdgeReg',
	definition = generateActivationEqnReg
)

setMethod(
  f = 'show',
  signature = 'EdgeReg',
  definition = function(object) {
    mxp = 10
    
    callNextMethod()
    cat('EC50:', object@EC50, '\n')
    cat('Hill constant (n):', object@n, '\n')
  }
)

#----GraphGRN----
setValidity('GraphGRN', validGraphGRN)

setMethod(
	f = 'initialize',
	signature = 'GraphGRN',
	definition = initGraphGRN
)

setMethod(
	f = 'show',
	signature = 'GraphGRN',
	definition = function(object) {
		mxp = 10
		nodes = object@nodeset
		edges = object@edgeset
		cat('GraphGRN object with', length(nodes), 'nodes and',
			length(edges), 'edges','\n')
		nodenames = sapply(nodes[1:min(mxp+1, length(nodes))], 
						   function(x) x$name)
		edgenames = sapply(edges[1:min(mxp+1, length(edges))], 
						   function(x) x$name)
		if (length(nodes) == 0) {
		  cat('Nodes: (0)', '\n')
		} else{
		  cat('Nodes:', maxPrint(nodenames, mxp, length(nodes)), '\n')
		}
		if (length(edges) == 0) {
		  cat('Edges: (0)', '\n')
		} else{
		  cat('Edges:', maxPrint(edgenames, mxp, length(edges)), '\n')
		}
	}
)

#----GraphGRN:addNode----
setGeneric(
	name = 'addNodeRNA',
	def = function(graph, node, tau, spmax, spdeg, logiceqn, inedges, outedges) {
		standardGeneric('addNodeRNA')
	}
)

setMethod(
	f = 'addNodeRNA',
	signature = c('GraphGRN', 'NodeRNA', 'missing', 'missing', 'missing', 'missing', 'missing', 'missing'),
	definition = function(graph, node, tau, spmax, spdeg, logiceqn, inedges, outedges) {
		graph@nodeset = c(graph@nodeset, node)
		
		#node name is not empty
		if (node$name %in% '') {
			stop('Node name cannot be empty')
		}
		
		#named entry to graph structure
		names(graph@nodeset)[length(graph@nodeset)] = node$name
		validObject(graph)
		return(graph)
	}
)

setMethod(
	f = 'addNodeRNA',
	signature = c('GraphGRN', 'character', 'ANY', 'ANY', 'ANY', 'ANY', 'missing', 'missing'),
	definition = function(graph, node, tau, spmax, spdeg, logiceqn, inedges, outedges) {
		#create default node
		nodeObj = new('NodeRNA', name = node)
		
		#modify default node with provided parameters
		if(!missing(tau) && !is.null(tau))
			edgeObj$tau = tau
		if(!missing(spmax) && !is.null(spmax))
			edgeObj$spmax = spmax
		if(!missing(spdeg) && !is.null(spdeg))
			edgeObj$spdeg = spdeg
		if(!missing(logiceqn) && !is.null(logiceqn))
			edgeObj$logiceqn = logiceqn
		
		graph = addNodeRNA(graph, nodeObj)
		
		return(graph)
	}
)

#----GraphGRN:removeNode----
setGeneric(
  name = 'removeNode',
  def = function(graph, nodenames) {
    standardGeneric('removeNode')
  }
)

setMethod(
  f = 'removeNode',
  signature = c('GraphGRN', 'character'),
  definition = function(graph, nodenames) {
    for (n in nodenames) {
      graph = rmnode(graph, n)
    }    
    return(graph)
  }
)

#----GraphGRN:addEdge----
setGeneric(
	name = 'addEdgeReg',
	def = function(graph, from, to, activation, weight, EC50, n) {
		standardGeneric('addEdgeReg')
	}
)

setMethod(
	f = 'addEdgeReg',
	signature = c('GraphGRN', 'character', 'character', 'ANY', 'ANY', 'ANY', 'ANY'),
	definition = function(graph, from, to, activation, weight, EC50, n) {
		#create default edge
		edgeObj = new('EdgeReg', from = from, to = to)
		#modify default edge with provided parameters
		if(!missing(activation) && !is.null(activation))
		  edgeObj$activation = activation
		if(!missing(weight) && !is.null(weight))
			edgeObj$weight = weight
		if(!missing(EC50) && !is.null(EC50))
			edgeObj$EC50 = EC50
		if(!missing(n) && !is.null(n))
			edgeObj$n = n
		
		#add edge to graph
		graph@edgeset = c(graph@edgeset, edgeObj)
		
		#update node inedges information
		tonode = graph@nodeset[[to]]
		tonode$inedges = c(tonode$inedges, edgeObj$name)
		#modify logic equation of target node
		fromeqn = from
		if (!edgeObj$activation) {
		  fromeqn = paste0('!', fromeqn)
		}
		if (is.na(tonode$logiceqn)) {
		  tonode$logiceqn = fromeqn
		} else{
		  tonode$logiceqn = paste(tonode$logiceqn, fromeqn, sep = ' & ')
		}
		#store modified node
		graph@nodeset[[tonode$name]] = tonode
		
		#update node outedges information
		fromnode = graph@nodeset[from]
		for (f in fromnode) {
		  f$outedges = c(f$outedges, edgeObj$name)
		  graph@nodeset[[f$name]] = f
		}
		
		#named entry to graph structure
		names(graph@edgeset)[length(graph@edgeset)] = edgeObj$name
		validObject(graph)
		
		return(graph)
	}
)

#----GraphGRN:removeEdge----
setGeneric(
  name = 'removeEdge',
  def = function(graph, from, to) {
    standardGeneric('removeEdge')
  }
)

setMethod(
  f = 'removeEdge',
  signature = c('GraphGRN', 'character', 'character'),
  definition = rmedge
)

#----GraphGRN:getEdge----
setGeneric(
	name = 'getEdge',
	def = function(graph, from, to) {
		standardGeneric('getEdge')
	}
)

setGeneric(
	name = 'getEdge<-',
	def = function(graph, from, to, value) {
		standardGeneric('getEdge<-')
	}
)

setMethod(
	f = 'getEdge',
	signature = c('GraphGRN', 'character', 'character'),
	definition = function(graph, from, to) {
		#generate name
		edgename = paste(sort(from), collapse = '')
		edgename = paste(edgename, to, sep = '->')
		
		edgeObj = graph@edgeset[[edgename]]
		return(edgeObj)
	}
)

setReplaceMethod(
	f = 'getEdge',
	signature = c('GraphGRN', 'character', 'character', 'Edge'),
	definition = function(graph, from, to, value) {
		#generate name
		edgename = paste(sort(from), collapse = '')
		edgename = paste(edgename, to, sep = '->')
		
		graph@edgeset[[edgename]] = value
		return(graph)
	}
)

#----GraphGRN:getInputNodes----
setGeneric(
	name = 'getInputNodes',
	def = function(graph) {
		standardGeneric('getInputNodes')
	}
)

setMethod(
	f = 'getInputNodes',
	signature = c('GraphGRN'),
	definition = function(graph) {
	  A = getAM(graph, directed = T)
	  #ignore self loops for identification of input nodes
	  diag(A) = 0
	  
	  #nodes with 0 degree
		inputnodes = colnames(A)[colSums(A) == 0]
		
		return(inputnodes)
	}
)

#----GraphGRN:getNode----
setGeneric(
	name = 'getNode',
	def = function(graph, nodename) {
		standardGeneric('getNode')
	}
)

setGeneric(
	name = 'getNode<-',
	def = function(graph, nodename, value) {
		standardGeneric('getNode<-')
	}
)

setMethod(
	f = 'getNode',
	signature = c('GraphGRN', 'character'),
	definition = function(graph, nodename) {
		nodeObj = graph@nodeset[[nodename]]
		return(nodeObj)
	}
)

setReplaceMethod(
	f = 'getNode',
	signature = c('GraphGRN', 'character', 'Node'),
	definition = function(graph, nodename, value) {
		graph@nodeset[[nodename]] = value
		return(graph)
	}
)

#----GraphGRN:nodenames----
setGeneric(
	name = 'nodenames',
	def = function(graph) {
		standardGeneric('nodenames')
	}
)

setMethod(
	f = 'nodenames',
	signature = c('GraphGRN'),
	definition = function(graph) {
		return(names(graph@nodeset))
	}
)

#----GraphGRN:edgenames----
setGeneric(
	name = 'edgenames',
	def = function(graph) {
		standardGeneric('edgenames')
	}
)

setMethod(
	f = 'edgenames',
	signature = c('GraphGRN'),
	definition = function(graph) {
		return(names(graph@edgeset))
	}
)

#----GraphGRN: generateODE----
setGeneric(
	name = 'generateODE',
	def = function(graph) {
		standardGeneric('generateODE')
	}
)

setMethod(
	f = 'generateODE',
	signature = c('GraphGRN'),
	definition = getODEFunc
)

#----GraphGRN: getSubGraph----
setGeneric(
  name = 'getSubGraph',
  def = function(graph, snodes) {
    standardGeneric('getSubGraph')
  }
)

setMethod(
  f = 'getSubGraph',
  signature = c('GraphGRN', 'character'),
  definition = subsetGraph
)

#----GraphGRN: getAM----
setGeneric(
  name = 'getAM',
  def = function(graph, directed) {
    standardGeneric('getAM')
  }
)

setMethod(
  f = 'getAM',
  signature = c('GraphGRN', 'logical'),
  definition = getAMC
)

setMethod(
  f = 'getAM',
  signature = c('GraphGRN', 'missing'),
  definition = function(graph, directed) {
    return(getAMC(graph, T))
  }
)

#----GraphGRN: sampleGraph----
setGeneric(
  name = 'sampleGraph',
  def = function(graph, size, minregs, k, seed) {
    standardGeneric('sampleGraph')
  }
)

setMethod(
  f = 'sampleGraph',
  signature = c('GraphGRN', 'numeric', 'ANY', 'ANY', 'ANY'),
  definition = function(graph, size, minregs, k, seed){
    if (missing(k))
      k = 0.25
    if (missing(seed))
      seed = sample.int(1E6, 1)
    if (missing(minregs))
      minregs = 0
    
    snodes = sampleSubNetwork(graph, size, minregs, k, seed)
    subgraph = getSubGraph(graph, snodes)
    return(subgraph)
  }
)

#----GraphGRN: randomizeParams----
setGeneric(
  name = 'randomizeParams',
  def = function(graph, type, seed) {
    standardGeneric('randomizeParams')
  }
)

setMethod(
  f = 'randomizeParams',
  signature = c('GraphGRN', 'character', 'numeric'),
  definition = randomizeParamsC
)


