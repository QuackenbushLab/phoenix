#----Node: functions----
validNode <- function(object) {
	#RNA maximum in range
	if (!is.na(object@spmax) & (object@spmax < 0 | object@spmax > 1)) {
		stop('RNA maximum expression has to be between 0 and 1')
	}
	
	#RNA degradation in range
	if (!is.na(object@spdeg) & (object@spdeg < 0 | object@spdeg > 1)) {
		stop('RNA degradation rate has to be between 0 and 1')
	}

  return(TRUE)
}

initNode <- function(.Object, ..., name = '', spmax = 1, spdeg = 1, logiceqn = as.character(NA), inedges = character(), outedges = character()) {
	.Object@name = name
	.Object@spmax = spmax
	.Object@spdeg = spdeg
	.Object@inedges = inedges
	.Object@outedges = outedges
	.Object@logiceqn = logiceqn
	
	validObject(.Object)
	return(.Object)
}

#----NodeRNA: functions----
validNodeRNA <- function(object){
	#Time constant in range
	if (!is.na(object@tau) & (object@tau <= 0)) {
		stop('Time constant must be positive')
	}
	
	return(TRUE)
}

initNodeRNA <- function(.Object, ..., tau = 1) {
	.Object@tau = tau
	.Object = callNextMethod()
	
	validObject(.Object)
	return(.Object)
}

#----NodeRNA: Eqn generation functions----
logiceqparser <- function(logiceq, node, graph) {
  estack = c()
  opstack = c()
  ops = c('&', '|', '+', '!', '(')
  precedence = c(4, 3, 2, 5, 1)
  fnnames = c('AND', 'OR', 'ADD', 'NOT', '')
  
  logiceq = str_replace_all(logiceq, ' ', '')
  while(!is.null(logiceq)) {
    if (grepl('^\\(', logiceq)) {
      opstack = c('(', opstack)
      logiceq = unlist(str_split(logiceq, '^\\(', n = 2))[2]
    } else if (grepl('^\\w+', logiceq)) {
      #expression stacked
      regname = str_extract(logiceq,'^\\w+') #extract regulator
      
      #check whether the node exists and whether the interaction exists
      if (!regname %in% nodenames(graph)) {
        stop(paste0('Node not found in graph: ', regname))
      }
      if (is.null(getEdge(graph, regname, node$name))) {
        stop(paste0('Edge not found: ', regname, '->', node$name))
      }
      
      regname = paste0('NODE(\'', regname, '\', node, graph)')
      estack = c(regname, estack)
      logiceq = unlist(str_split(logiceq, '^\\w+', n = 2))[2]
    } else if (grepl('^[\\+\\|&!]{1}', logiceq)) {
      op = str_extract(logiceq,'^[\\+\\|&!]{1}')
      #perform operations if precendence is higher or same
      while (length(opstack) != 0 && precedence[which(ops %in% opstack[1])] >= precedence[which(ops %in% op)]) {
        if (opstack[1] %in% '!'){
          e1 = estack[1]
          estack = estack[-1]#pop expr
          estack = c(paste0(fnnames[which(ops %in% opstack[1])], '(', e1, ')'), estack)
          opstack = opstack[-1]#pop op
        } else if (opstack[1] %in% c('&', '|')){
          e2 = estack[1]
          e1 = estack[2]
          estack = estack[-(1:2)]#pop exprs
          estack = c(paste0(fnnames[which(ops %in% opstack[1])], '(', e1, ', ', e2, ')'), estack)
          opstack = opstack[-1]#popop
        } else{
          if (op %in% '+'){
            break
          } else{
            numops = rle(opstack)$lengths[1]
            es = estack[1:(numops + 1)]
            estack = estack[-(1:(numops + 1))]#pop exprs
            estack = c(paste0('ADD', '(', paste(es, collapse = ', '), ')'), estack)
            opstack = opstack[-(1:numops)]#popop
          }
        }
      }
      
      #operation stacked
      opstack = c(op, opstack)
      logiceq = unlist(str_split(logiceq, '^[\\+\\|&!]{1}', n = 2))[2]#digest eq
    } else if (grepl('^\\)', logiceq)) {
      #end of brackets
      while (opstack[1] != '(') {
        if (opstack[1] %in% '!'){
          e1 = estack[1]
          estack = estack[-1]#pop expr
          estack = c(paste0(fnnames[which(ops %in% opstack[1])], '(', e1, ')'), estack)
          opstack = opstack[-1]#pop op
        } else if (opstack[1] %in% c('&', '|')){
          e2 = estack[1]
          e1 = estack[2]
          estack = estack[-(1:2)]#pop exprs
          estack = c(paste0(fnnames[which(ops %in% opstack[1])], '(', e1, ', ', e2, ')'), estack)
          opstack = opstack[-1]#popop
        } else{
          numops = rle(opstack)$lengths[1]
          es = estack[1:(numops + 1)]
          estack = estack[-(1:(numops + 1))]#pop exprs
          estack = c(paste0('ADD', '(', paste(es, collapse = ', '), ')'), estack)
          opstack = opstack[-(1:numops)]#popop
        }
      }
      
      #operation stacked
      opstack = opstack[-1]
      logiceq = unlist(str_split(logiceq, '^\\)', n = 2))[2]
    } else if (logiceq == '') {
      #end of equation
      while (length(opstack) != 0) {
        if (opstack[1] %in% '!'){
          e1 = estack[1]
          estack = estack[-1]#pop expr
          estack = c(paste0(fnnames[which(ops %in% opstack[1])], '(', e1, ')'), estack)
          opstack = opstack[-1]#pop op
        } else if (opstack[1] %in% c('&', '|')){
          e2 = estack[1]
          e1 = estack[2]
          estack = estack[-(1:2)]#pop exprs
          estack = c(paste0(fnnames[which(ops %in% opstack[1])], '(', e1, ', ', e2, ')'), estack)
          opstack = opstack[-1]#popop
        } else{
          numops = rle(opstack)$lengths[1]
          es = estack[1:(numops + 1)]
          estack = estack[-(1:(numops + 1))]#pop exprs
          estack = c(paste0('ADD', '(', paste(es, collapse = ', '), ')'), estack)
          opstack = opstack[-(1:numops)]#popop
        }
      }
      
      logiceq = NULL
    } else{
      stop(paste0('Unexpected symbol at: ', logiceq))
    }
  }
  
  return(estack)
}

NODE <- function(expr, node, graph){
  e = getEdge(graph, expr, node@name)
  expr = generateActivationEqn(e)
  expr = paste(e@weight, expr, sep = ' * ')
  
  return(expr)
}

NOT <- function(expr) {
  expr = paste0('(1 - ', expr, ')')
  return(expr)
}

AND <- function(expr1, expr2) {
  expr = paste0('(', expr1, ' * ', expr2, ')')
  return(expr)
}


OR <- function(expr1, expr2) {
  expr = paste0('(', expr1, ' + ', expr2, ' - ', expr1, ' * ', expr2, ')')
  return(expr)
}

#OR <- function(expr1, expr2) {
# #browser()
#  expr = paste0('(', expr1, ' * ', expr2, ')')
#  return(expr)
#}

ADD <- function(...) {
  exprs = c(...)
  exprs = paste(paste0('1/',length(exprs)), exprs, sep = ' * ')
  expr = paste0('(', paste(exprs, collapse = ' + '), ')')
  return(expr)
}

generateRateEqn <- function(node, graph) {
 
  #no rate equations for input nodes
  if (is.na(node@logiceqn)) {
    return('')
  }
  
  logiceqnR = logiceqparser(node@logiceqn, node, graph)
  act = with(list(node, graph), eval(parse(text = logiceqnR)))
 # act = paste('addBioNoise(', act, ')', sep = '')
	
	#generate rate equation
	rateEqn = paste(act, node@spmax, sep = ' * ')
	degradationEqn = paste(node@spdeg, 
	                       node@name,
	                       sep = ' * ')
	                       
	rateEqn = paste(rateEqn, degradationEqn, sep = ' - ')
	rateEqn = paste('(', rateEqn, ') / ', node@tau, sep = '')
	return(rateEqn)
}

generateRateEqn_modified <- function(node, graph) {
  #no rate equations for input nodes
  if (is.na(node@logiceqn)) {
    return('')
  }
  
  logiceqnR = logiceqparser(node@logiceqn, node, graph)
  act = with(list(node, graph), eval(parse(text = logiceqnR)))
  act = paste('(', act, ')', sep = '')
  
  #generate rate equation
  rateEqn = paste(act, node@spmax, sep = ' * ')
  degradationEqn = paste(node@spdeg, node@name, sep = ' * ')
  rateEqn = paste(rateEqn, degradationEqn, sep = ' - ')
  rateEqn = paste('(', rateEqn, ') / ', node@tau, sep = '')
  return(rateEqn)
}


#----Edge: functions----
validEdge <- function(object) {
  #Number of regulators is 1
  if (length(object@to) != 1) {
    stop('Only 1 target node should be provided for interactions')
  }
  
	#weight in range
	if (any(is.na(object@weight)) | sum(object@weight < 0 | object@weight > 1) > 0) {
		stop('Interaction weight has to be between 0 and 1')
	}
}

validActivationParams <- function(object) {
	#EC50 in range
	if (any(is.na(object@EC50)) | sum(object@EC50 < 0 | object@EC50 > 1) > 0) {
		stop('EC50 has to be between 0 and 1')
	}
	
	#Hill constant in range
	if (any(is.na(object@n)) | sum(object@n == 1) > 0) {
		stop('Hill constant (n) cannot be 1')
	}
	
	return(TRUE)
}

#----EdgeReg: functions----
validEdgeReg <- function(object) {
  validActivationParams(object)
  
	#Weight length is 1
	if (length(object@weight) != 1) {
		stop('Only 1 parameter for the weight should be provided for OR interactions')
	}
	
	#EC50 length is 1
	if (length(object@EC50) != 1) {
		stop('Only 1 parameter for the EC50 should be provided for OR interactions')
	}
	
	#Hill constant length is 1
	if (length(object@n) != 1) {
		stop('Only 1 parameter for the Hill constant(n) should be provided for OR interactions')
	}
	
	#Number of regulators is 1
	if (length(object@from) != 1) {
		stop('Only 1 source node should be provided for regulatory interactions')
	}
	
	return(TRUE)
}

initEdgeReg <- function(.Object, ..., from, to, weight = 1, EC50 = 0.5, n = 1.39, activation = T) {
  .Object@from = from
  .Object@to = to
  .Object@weight = weight
  .Object@EC50 = EC50
  .Object@n = n
  .Object@activation = activation
  
  #generate name
  name = paste(sort(from), collapse = '')
  name = paste(name, to, sep = '->')
  .Object@name = name
  
  validObject(.Object)
  return(.Object)
}

generateActivationEqnReg <- function(object) {
	e = object
	#generate activation eqn
	act = paste('fAct(', e$from, ', ', e$EC50, ', ', e$n, ')', sep =	'')
	
	return(act)
}

#----GraphGRN: core functions----
validGraphGRN <- function(object) {
	#nodeset are all of class Node
	if (!all(sapply(object@nodeset, is, 'Node'))) {
		stop('All nodes must be of class \'Node\'')
	}
	
	#check names of nodeset
	if (!all(sapply(object@nodeset, function(x) x$name) == names(object@nodeset))){
		stop('Invalid graph generated. Use the \'addNode\' method to add a node to the graph.')
	}
	
	#edgeset are all of class Edge
	if (!all(sapply(object@edgeset, is, 'Edge'))) {
		stop('All nodes must be of class \'Edge\'')
	}
	
	#check names of edgeset
	if (!all(sapply(object@edgeset, function(x) x$name) == names(object@edgeset))){
		stop('Invalid graph generated. Use the \'addEdge\' method to add an edge to the graph.')
	}
  
  #edge checks
  nnames = names(object@nodeset)
  for (e in object@edgeset) {
    if (!e$from %in% nnames) {
      stop('Source nodes do not exist in edge: ', e$name)
    }
    if (!e$to %in% nnames) {
      stop('Target node does not exist in edge: ', e$name)
    }
  }
  
  #node checks
  #check node duplication
  if (length(unique(nnames)) != length(nnames)) {
    stop('All nodes in the graph must be unique')
  }
  
  enames = names(object@edgeset)
  for (n in object@nodeset) {
    #check whether the incoming and outgoing edges exist
    if (!all(n$inedges %in% enames)) {
      stop('Some/all inbound edges do not exist in node: ', n$name)
    }
    if (!all(n$outedges %in% enames)) {
      stop('Some/all outbound edges do not exist in node: ', n$name)
    }
    
    #Incoming edges: check that all edges have to as this node
    infroms = sapply(object@edgeset[n$inedges], function(e) e$from)
    intos = sapply(object@edgeset[n$inedges], function(e) e$to)
    
    if (!all(intos %in% n$name)) {
      stop('All inbound interactions must have the \'target\' node as current node for node: ', n$name)
    }
    
    #Incoming edges: check that all regulators are unique
    if (length(unique(infroms)) != length(infroms)) {
      stop('All regulators must be unique for node: ', n$name)
    }
    
    #Outgoing edges: check that all edges have from as this node
    outfroms = sapply(object@edgeset[n$outedges], function(e) e$from)
    outtos = sapply(object@edgeset[n$outedges], function(e) e$to)
    
    if (!all(outfroms %in% n$name)) {
      stop('All oubdound interactions must have the current node as a \'source\' node: ', n$name)
    }
    
    #Outgoing edges: check that all targets are unique
    if (length(unique(outtos)) != length(outtos)) {
      stop('All targets must be unique for node: ', n$name)
    }
  }
	
	return(TRUE)
}

initGraphGRN <- function(.Object, ..., nodeset = list(), edgeset = list()) {
	.Object@nodeset = nodeset
	.Object@edgeset = edgeset
	
	validObject(.Object)
	
	return(.Object)
}

removeMissing <- function(graph) {
  #remove nodes with no interactions
  intnodes = unlist(sapply(graph@edgeset, function (e) e$from))
  intnodes = c(intnodes, sapply(graph@edgeset, function (e) e$to))
  intnodes = unique(intnodes)
  
  #warning and remove
  nonintnodes = setdiff(names(graph@nodeset), intnodes)
  if (length(nonintnodes) != 0) {
    msg = paste0('Nodes without interactions removed: ', paste(nonintnodes, collapse = ', '))
    warning(msg)
    
    graph = removeNode(graph, nonintnodes)
  }
  return(graph)
}

#----GraphGRN: specific functions----
getODEFunc <- function(graph) {
  graph = removeMissing(graph)
  fn = 'function(t, state, parameters) {'
  
  #define the activation function
  fn = paste(fn, '\tfAct <- function(TF, EC50 = 0.5, n = 1.39) {', sep = '\n')
  fn = paste(fn, '\t\tB = (EC50 ^ n - 1) / (2 * EC50 ^ n - 1)', sep = '\n')
  fn = paste(fn, '\t\tK_n = (B - 1)', sep = '\n')
  fn = paste(fn, '\t\tact = B * TF ^ n / (K_n + TF ^ n)', sep = '\n')
  fn = paste(fn, '\t\t', sep = '\n')
  fn = paste(fn, '\t\treturn(act)', sep = '\n')
  fn = paste(fn, '\t}', sep = '\n')
  fn = paste(fn, '\t', sep = '\n')
  
  #function body
  fn = paste(fn, '\tparms = c(list(), parameters, state)', sep = '\n')
  fn = paste(fn, '\trates = state * 0', sep = '\n')
  fn = paste(fn, '\trates = with(parms, {', sep = '\n')
  #start with: create equations
  inputNodes = getInputNodes(graph)
  counter = 0
  for (node in graph@nodeset) {
    if(node$name %in% inputNodes)
      next
    counter = counter + 1
    eqn = paste('\t\t', 'rates[', counter, '] = ', generateRateEqn(node, graph), sep = '')
    fn = paste(fn, eqn, sep = '\n')
  }
  
  #end with
  fn = paste(fn, '\t\treturn(rates)', sep = '\n')
  fn = paste(fn, '\t})', sep = '\n')
  
  #end function
  fn = paste(fn, '\n\treturn(list(rates))', sep = '\n')
  fn = paste(fn, '}', sep = '\n')
  
  return(eval(parse(text = fn)))
}


#----GraphGRN: specific functions----
getODEFunc_modified <- function(graph) {
  graph = removeMissing(graph)
  rates = list()
  #start with: create equations
  inputNodes = getInputNodes(graph)
  rates <- lapply(graph@nodeset, function(node){
    if(!node$name %in% inputNodes){
      return(list(node = node$name,
                  eqn = generateRateEqn_modified(node, graph)))
    }else{
      return(list(node = node$name,
                  eqn = "input gene"))
    }
        
  })
  
  
  return(rbindlist(rates))
}

addBioNoise <- function(x, lnorm){
  if (all(lnorm == 1)) {
    return(x)
  }
  
  #transformation fn
  f <- function(a, b){
    fnval = a * exp(-0.01 *( 1 - a/(1 - a))) - b
    return(fnval)
  }
  optf <- function(a, b){
    fnval = a * exp(-0.01 *( 1 - a/(1 - a))) - b
    return(abs(fnval))
  }
  
  #transformation
  y = x
  tfx = x>=0.65
  y[tfx] = f(x[tfx], 0)
  y[!tfx] = x[!tfx]
  
  #apply lognormal noise
  newy = y * lnorm
  
  #inverse transformation
  newx = x
  tfy = newy>=0.65
  newx[tfy] = unlist(sapply(newy[tfy], function (a)
    optim(
      runif(1),
      optf,
      b = a,
      lower = 0,
      upper = 1,
      method = 'Brent',
      control = list('abstol' = 1E-8)
    )$par))
  newx[!tfy] = newy[!tfy]
  return(newx)
}

rmnode <- function(graph, nodename) {
  #Check nodename exists
  if (!nodename %in% names(graph@nodeset)) {
    stop('Node not found')
  }
  
  #retrieve node
  node = getNode(graph, nodename)
  
  #remove from inedges of targets
  edges = graph@edgeset[c(node$inedges, node$outedges)]
  for (e in edges) {
    graph = removeEdge(graph, e$from, e$to) #remove edge
  }

  #remove node from nodeset
  graph@nodeset = graph@nodeset[!names(graph@nodeset) %in% node$name]
  
  return(graph)
}

rmedge <- function(graph, from, to) {
  #ensure or edges exists
  if (length(to) > 1)
    stop('Multiple target nodes provided, expected 1')
  if (length(from) > 1)
    stop('Multiple source nodes provided, expected 1')
  
  edge = getEdge(graph, from, to)
  from = getNode(graph, from)
  to = getNode(graph, to)
  if (is.null(edge)) {
    stop('Edge not found')
  }
  
  #remove edge from outedges list of source node
  from$outedges = setdiff(from$outedges, edge$name)
  
  #remove edge from inedges list of target node
  to$inedges = setdiff(to$inedges, edge$name)
  
  #modify logic equation of target node
  logiceq = str_replace_all(to$logiceqn, ' ', '')
  
  ops = c('&', '|', '+', '!', '(', ')', '')
  precedence = c(4, 3, 2, 5, 1, 1, 0)
  
  nn = paste0('!?', '\\b', from$name, '\\b')
  nn = paste0('(', nn, '|\\(', nn, '\\))')
  opregexp = '[\\(\\)\\+\\|&]?'
  regexp = paste0(opregexp, nn, opregexp)
  expr = unlist(str_extract_all(logiceq, regexp))
  exprops = unlist(str_split(expr, nn))
  if (precedence[ops %in% exprops[2]] >= precedence[ops %in% exprops[1]]) {
    newop = exprops[1]
  } else{
    newop = exprops[2]
  }
  
  logiceq = gsub(regexp, newop, logiceq)
  if (logiceq %in% '') {
    logiceq = as.character(NA)
  }
  to$logiceqn = logiceq
  
  #remove edge from edgeset of graph
  graph@edgeset = graph@edgeset[!names(graph@edgeset) %in% edge$name]
  getNode(graph, to$name) = to
  getNode(graph, from$name) = from
  
  return(graph)
}

subsetGraph <- function(graph, snodes) {
  #remove nodes from the graph
  for (n in setdiff(names(graph@nodeset), snodes)) {
    graph = removeNode(graph, n)
  }
  return(graph)
}

sampleSubNetwork <- function(graph, size, minregs, k, seed) {
  #identify regulators
  regs = rowSums(getAM(graph))
  regs = names(regs)[regs > 0]
  
  #get the adjacency matrix for the graph
  A = getAM(graph, directed = F)
  hdegree = rowSums(A)
  hdegree = names(hdegree)[hdegree > 1]
  
  #calculate total number of edges
  m = sum(diag(A)) + sum(A[upper.tri(A)])
  
  #calculate theoretical edge numbers between nodes
  degrees = rowSums(A)
  P = (degrees %*% t(degrees)) / (2 * m)
  B = A - P
  
  #start with empty network
  s = rep(-1, ncol(A)) # 1 or -1 for inc. or excl. resp.
  
  #set seed and start adding neighbours
  set.seed(seed)
  s[sample(which(s < 0), 1)] = 1
  
  for (i in 2:size) {
    #find neighbours
    neighbours = which(colSums(A[s > 0, , drop = F]) > 0 & s < 0)
    
    #sample the minimum number of regulators required
    if (minregs > 0) {
      newn = intersect(names(neighbours), regs)
      newnhdeg = intersect(names(neighbours), hdegree)
      
      if (length(newn) != 0) {
        neighbours = neighbours[newn]
        minregs = minregs - 1
      } else if (length(newnhdeg) != 0) {
        #favour higher degree neighbours to improve chances of hitting a reg
        neighbours = neighbours[newnhdeg]
      }
    }
    
    if (length(neighbours) == 0) {
      s[sample(which(s < 0), 1)] = 1
      next
    }
    
    #generate subnetworks
    subs = s %*% t(rep(1, length(s)))
    diag(subs) = 1
    subs = subs[ , neighbours]
    #calculate modulatiry
    Q = diag((t(subs) %*% B %*% subs) / (4 * m))
    cand = neighbours[Q >= quantile(Q, 1-k)] #candidates for addition
    s[cand[sample.int(length(cand),1)]] = 1
  }
  
  #return names of nodes in sampled subgraph
  return(colnames(A)[s > 0])
}

getAMC <- function(graph, directed = T) {
  nodes = graph@nodeset
  edges = graph@edgeset
  
  A = matrix(rep(0, length(nodes) ^ 2), nrow = length(nodes))
  colnames(A) = rownames(A) = names(nodes)
  
  #generate adjacency matrix: true edge numbers between nodes
  for (e in edges) {
    A[e$from, e$to] = 1
  }
  
  #if undirected
  if (!directed) {
    A = apply(A, 2, as.logical)
    A = A | t(A)
    A = apply(A, 2, as.numeric)
    rownames(A) = colnames(A)
  }
  
  return(A)
}

randomizeParamsC <- function(graph, type = 'linear', seed) {
  ec50range = c(0.2, 0.8)
  nrange = c(1.39, 6)
  if (type %in% 'linear') {
    ec50range = c(0.5, 0.5)
    nrange = c(1.01, 1.01)
  } else if (type %in% 'linear-like') {
    ec50range = c(0.4, 0.6)
    nrange = c(1.39, 1.8)
    #nrange = c(1.01, 1.7)
  } else if (type %in% 'exponential') {
    ec50range = c(0.2, 0.8)
    nrange = c(1.01, 1.7)
  } else if (type %in% 'sigmoidal') {
    ec50range = c(0.4, 0.6)
    nrange = c(2, 6)
  } else if (type %in% 'mixed') {
    ec50range = c(0.2, 0.8)
    nrange = c(1.01, 6)
  } else{
    stop('Unknown type specified. Possible types: linear, linear-like, exponential, sigmoidal, mixed')
  }
  
  #modify parameters
  for (e in graph@edgeset) {
    getEdge(graph, e$from, e$to)$EC50 = runif(1, ec50range[1], ec50range[2])
    getEdge(graph, e$from, e$to)$n = runif(1, nrange[1], nrange[2])
  }
  
  return(graph)
}

genericLogicEqn <- function(node, graph, propor = 0.1, outdegree = NULL) {
  if (is.null(outdegree)) {
    am = getAM(graph)
    outdegree = rowSums(am)
  }
  
  #get regulators, activation status and outdegree
  regs = graph@edgeset[node$inedges]
  regact = sapply(regs, slot, 'activation')
  regs = sapply(regs, slot, 'from')
  
  if (length(regs) == 0)
    return(as.character(NA))
  regdegree = outdegree[regs]
  
  tf = regs[regdegree == max(regdegree)][1]
  
  if (regact[regs %in% tf]) {
    #tf is an activator
    coacts = regs[!regs %in% tf & regact == T]
    coreps = setdiff(regs, c(tf, coacts))
    
    logiceqn = tf
    if (length(coacts) > 0){
      actands = rep(' & ', length(coacts) - 1)
      actands[runif(length(actands)) < propor] = ' | '
      actands = c(actands, '')
      coacteqn = paste(paste(coacts, actands, sep = ''), collapse = '')
      if (length(coacts) == 1) {
        logiceqn = paste(logiceqn, ' & ', coacteqn, sep = '')
      } else{
        logiceqn = paste(logiceqn, ' & (', coacteqn, ')', sep = '')
      }
    }
    
    if (length(coreps) > 0){
      repands = rep(' | ', length(coreps) - 1)
      repands[runif(length(repands)) < propor] = ' & '
      repands = c(repands, '')
      corepeqn = paste(paste('!', coreps, repands, sep = ''), collapse = '')
      if (length(coreps) == 1) {
        logiceqn = paste(logiceqn, ' & ', corepeqn, sep = '')
      } else{
        logiceqn = paste(logiceqn, ' & (', corepeqn, ')', sep = '')
      }
    }
  } else{
    #tf is a repressor
    coreps = regs[!regs %in% tf & regact == F]
    coacts = setdiff(regs, c(tf, coreps))
    
    logiceqn = paste('!', tf, sep = '')
    if (length(coacts) > 0){
      actands = rep(' & ', length(coacts) - 1)
      actands[runif(length(actands)) < propor] = ' | '
      actands = c(actands, '')
      coacteqn = paste(paste(coacts, actands, sep = ''), collapse = '')
      if (length(coacts) == 1) {
        logiceqn = paste(logiceqn, ' | ', coacteqn, sep = '')
      } else{
        logiceqn = paste(logiceqn, ' | (', coacteqn, ')', sep = '')
      }
    }
    
    if (length(coreps) > 0){
      repands = rep(' | ', length(coreps) - 1)
      repands[runif(length(repands)) < propor] = ' & '
      repands = c(repands, '')
      corepeqn = paste(paste('!', coreps, repands, sep = ''), collapse = '')
      if (length(coreps) == 1) {
        logiceqn = paste(logiceqn, ' | ', corepeqn, sep = '')
      } else{
        logiceqn = paste(logiceqn, ' | (', corepeqn, ')', sep = '')
      }
    }
  }
  
  return(logiceqn)
}

#----GraphGRN: Conversion functions----
df2GraphGRN <- function(edges, nodes, propor = 0, loops = F, seed = sample.int(1E6, 1)) {
  if (missing(nodes) || is.null(nodes)) {
    nodes = data.frame('node' = unique(c(edges[ , 1], edges[ , 3])), stringsAsFactors = F)
  }
  
  #remove loops if required
  if (!loops) {
    edges = edges[edges[, 1] != edges[, 3], ]
  }
  
  #names for the main columns should be consistent
  colnames(edges)[1:3] = c('from', 'activation', 'to')
  colnames(nodes)[1] = c('node')
  
  #create graph object
  grn = new('GraphGRN')
  #add nodes
  tot_nodes = nrow(nodes)
  for (i in 1:tot_nodes) {
    print(paste(i, "of", tot_nodes, "nodes" ))
    n = nodes[i, , drop = F]
    grn = addNodeRNA(grn, n$node, n$tau, n$max, n$deg)
  }
  
  #add edges
  tot_edges = nrow(edges)
  for (i in 1:tot_edges) {
    print(paste(i, "of", tot_edges, "edges" ))
    e = edges[i, , drop = F]
    grn = addEdgeReg(grn, e$from, e$to, e$activation, e$weight, e$EC50, e$n)
  }
  
  #create generic equations with random interactions
  set.seed(seed)
  am = getAM(grn)
  outdegree = rowSums(am)
  size_nodeset = length(grn@nodeset)
  count = 0
  for (n in grn@nodeset) {
    count = count + 1
    print(paste(count, "of", size_nodeset, "node set" ))
    logiceqn = genericLogicEqn(n, grn, propor, outdegree)
    getNode(grn, n$name)$logiceqn = logiceqn
  }
  
  return(grn)
}

GraphGRN2df <- function(graph) {
  nodes = graph@nodeset
  edges = graph@edgeset
  
  #create node df
  nodedf = data.frame('name' = names(nodes), stringsAsFactors = F)
  nodedf$rnamax = sapply(nodes, slot, 'spmax')
  nodedf$rnadeg = sapply(nodes, slot, 'spdeg')
  nodedf$tau = sapply(nodes, slot, 'tau')
  nodedf$logiceqn = sapply(nodes, slot, 'logiceqn')
  nodedf$type = sapply(nodes, class)
  
  #create edge df
  #convert oredges
  if (length(edges) == 0) {
    edgedf = data.frame('from' = as.numeric(), stringsAsFactors = F)
  } else{
    edgedf = data.frame('from' = sapply(edges, slot, 'from'), stringsAsFactors = F)
  }
  edgedf$to = sapply(edges, slot, 'to')
  edgedf$weight = sapply(edges, slot, 'weight')
  edgedf$EC50 = sapply(edges, slot, 'EC50')
  edgedf$n = sapply(edges, slot, 'n')
  edgedf$activation = sapply(edges, slot, 'activation')
  edgedf$type = sapply(edges, class)
  
  #convert types
  for (i in 2:4){
    nodedf[ , i] = as.numeric(nodedf[ , i])
  }
  
  for (i in 3:5){
    edgedf[ , i] = as.numeric(edgedf[ , i])
  }
  
  #create list of results
  rownames(nodedf) = NULL
  rownames(edgedf) = NULL
  dflist = list('nodes' = nodedf, 'edges' = edgedf)
  
  return(dflist)
}

GraphGRN2igraph <- function(graph, directed = T) {
  dflist = GraphGRN2df(graph)
  ig = graph_from_data_frame(d = dflist$edges, directed = directed, vertices = dflist$nodes)
  
  return(ig)
}

#----All classes----
maxPrint <- function(charvec, maxprint = 10, len = NULL) {
	if(is.null(len)){
		len = length(charvec)
	}
	txt = paste0('(', len, ')')
	
	if (length(charvec) > maxprint) {
		charvec = c(charvec[1:maxprint],  '...')
	}
	txt = paste(txt, paste(charvec, collapse = ', '))
	return(txt)
}

