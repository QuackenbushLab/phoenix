addNormNoise <- function(vec, noise){
  vec + rnorm(length(vec),mean = 0, sd = noise)
}

#----Simulation: functions----
validSimulationGRN <- function(object) {
	#graph is a GraphGRN object
	if (!is(object@graph, 'GraphGRN')) {
		stop('graph must be a GraphGRN object')
	}
	
	#local noise
	if (object@expnoise<0) {
		stop('Experimental noise standard deviation must be greater than 0')
	}
	
	#global noise
	if (object@bionoise<0) {
		stop('Biological noise standard deviation must be greater than 0')
	}
	
	return(TRUE)
}

initSimulationGRN <- function(.Object, ..., graph, expnoise = 0, bionoise = 0, seed = sample.int(1e6,1), inputModels = list(), propBimodal = 0) {
	.Object@graph = graph
	.Object@expnoise = expnoise
	.Object@bionoise = bionoise
	.Object@seed = seed
	.Object@inputModels = inputModels
	
	if (length(inputModels) == 0) {
	  .Object = generateInputModels(.Object, propBimodal)
	}
	
	validObject(.Object)
	return(.Object)
}


createInputModels <- function(simulation, propBimodal) {
  set.seed(simulation@seed)
  
  #create input models
  innodes = getInputNodes(simulation@graph)
  inmodels = list()
  
#  browser()
  for (n in innodes) {
   # browser()
    parms = list()
    mxs = sample(c(1, 2), 1, prob = c(1 - propBimodal, propBimodal))
    
    if (mxs == 2) {
      parms = c(parms, 'prop' = runif(1, 0.2, 0.8))
      parms$prop = c(parms$prop, 1 - parms$prop)
      parms$mean = c(rbeta(1, 10, 100), rbeta(1, 10, 10))
    } else {
      parms$prop = 1
      parms$mean = rbeta(1, 10, 10)
    }
    
    maxsd = pmin(parms$mean, 1 - parms$mean) / 3
    parms$sd = sapply(maxsd, function(x) max(rbeta(1, 15, 15) * x, 0.01))
    inmodels = c(inmodels, list(parms))
  }
  
  names(inmodels) = innodes
  simulation@inputModels = inmodels
  
  return(simulation)
}

#src = https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
#src = Lewandowski, Kurowicka, and Joe (LKJ), 2009
#lower betaparam gives higher correlations
vineS <- function(d, betaparam = 5, seed = sample.int(1E6, 1)) {
  set.seed(seed)
  P = matrix(rep(0, d ^ 2), ncol = d)
  S = diag(rep(1, d))
  
  for (k in 2:(d - 1)) {
    for (i in (k + 1):d) {
      P[k, i] = rbeta(1, betaparam, betaparam)
      P[k, i] = (P[k, i] - 0.5) * 2
      p = P[k, i]
      for (l in (k - 1):1) {
        p = p * sqrt((1 - P[l, i] ^ 2) * (1 - P[l, k] ^ 2)) + P[l, i] * P[l, k]
      }
      S[k, i] = p
      S[i, k] = p
    }
  }
  
  permutation = sample(1:d, d)
  S = S[permutation, permutation]
  
  return(S)
}

# beta = 0 means no correlated inputs, smaller beta means stronger correlations
generateInputData <- function(simulation, numsamples, cor.strength = 5, inputGeneVar) {
  set.seed(simulation@seed)
  
  innodes = getInputNodes(simulation@graph)
  externalInputs = matrix(-1,nrow = numsamples, ncol = length(innodes))
  colnames(externalInputs) = innodes
  
  #create input models
  if (length(simulation@inputModels) == 0) {
    simulation = generateInputModels(simulation)
  }
  
  #simulate external inputs
  inmodels = simulation@inputModels
  classf = c()
  for (n in innodes) {
    m = inmodels[[n]]
    mix = sample(1:length(m$prop), numsamples, prob = m$prop, replace = T)
    
    outbounds = 1
    while (sum(outbounds) > 0){
      outbounds = externalInputs[ , n] < 0 | externalInputs[ , n] > 1
      externalInputs[outbounds & mix == 1, n] = rnorm(sum(outbounds & mix == 1), m$mean[1], m$sd[1] * inputGeneVar)
      if (length(m$prop) > 1) {
        externalInputs[outbounds & mix == 2, n] = rnorm(sum(outbounds & mix == 2), m$mean[2], m$sd[2] * inputGeneVar)
      }
    }
    
    if (length(m$prop) > 1) {
      #save class information
      classf = rbind(classf, mix)
      rownames(classf)[nrow(classf)] = n
    }
  }
  
  #correlated inputs
  if (cor.strength > 0 & numsamples > 1) {
    inputs = ncol(externalInputs)
    dm = apply(externalInputs, 2, sort)
    covmat = vineS(inputs, cor.strength, simulation@seed)
    cordata = mvrnorm(numsamples, rep(0, inputs), covmat)
    for (i in 1:inputs) {
      #avoid correlated bimodal inputs
      if (i %in% which(innodes %in% rownames(classf))) {
        cordata[, i] = externalInputs[, i]
      } else {
        cordata[, i] = dm[, i][rank(cordata[, i])]
      }
    }
    
    externalInputs = cordata
  }
  
 
  
  #add mixture info to attributes
  attr(externalInputs, 'classf') = classf
  #browser()
  colnames(externalInputs) = innodes
  return(externalInputs)
}

#cor.strength used for generating correlated inputs
simDataset <- function(simulation, numsamples, cor.strength, externalInputs,timeStamps, inputGeneVar, outputGeneVar) {
  #browser()
  if (missing(cor.strength)) {
    cor.strength = 5
  }
  
  #generate input matrix
  innodes = getInputNodes(simulation@graph)
  if (!missing(externalInputs) && !is.null(externalInputs)) {
    if (nrow(externalInputs) != numsamples |
        length(setdiff(innodes, colnames(externalInputs))) != 0) {
          stop('Invalid externalInputs matrix provided')
    }
    externalInputs = externalInputs[, innodes, drop = F]
    classf = NULL
  } else{
    #browser()
    externalInputs = generateInputData(simulation, numsamples, cor.strength, inputGeneVar)
    
    #extract class information
    classf = attr(externalInputs, 'classf')
  }
  
  #set random seed
  set.seed(simulation@seed)
  
  #solve ODE
  graph = simulation@graph
  my_ode = generateODE(graph)
  
  #generate LN noise for simulation
  #lnnoise = exp(rnorm(numsamples * length(nodenames(graph)), 0, simulation@bionoise))
  #lnnoise = matrix(lnnoise, nrow = numsamples, byrow = T)
  #colnames(lnnoise) = nodenames(graph)
  
  #browser()
  #initialize solutions
  nodes = setdiff(nodenames(graph), colnames(externalInputs))
  exprs = matrix(NA,ncol = length(nodes), nrow = numsamples)
  
  #initialize output genes
  #every output gene has a slightly different mean, but we control the overall SD
  for(col in 1:length(nodes)){
    exprs[,col] <- rbeta(numsamples, 2 * 1/outputGeneVar, 2* 1/outputGeneVar)
    mean_pos <- runif(1,min = -0.25, max = 0.25)
    exprs[,col] <- exprs[,col] + mean_pos
  }
  exprs[exprs < 0] = 0
  exprs[exprs > 1] = 1
  
  colnames(exprs) = nodes

  #solve ODEs for different inputs
  res <- lapply(1:numsamples, function(i){
    message(paste("solving",i,"of",numsamples))
    times <- timeStamps
    
    
    get_derivative_instead = FALSE
    if(get_derivative_instead){
      print("getting derivative instead!")
      basic_soln <- deSolve::ode(y = exprs[i, ], 
                                 times = times,
                                 func = my_ode, 
                                 parms = externalInputs[i, ])
      basic_soln_DT <- data.table( basic_soln)
      this_deriv <- lapply(times, function(this_time){
        my_ode( t = this_time, 
                state = as.matrix(basic_soln_DT[time == this_time, -1, with = F])[1,], 
                parameters =  externalInputs[i, ])[[1]]
      }) 
      soln_DT <- data.table(do.call(rbind, this_deriv))
      soln_DT[, time := times]
      soln_DT[,sample:=i ]
      input_col_names <- paste(names(externalInputs[i, ]),"input", sep = "_")
      soln_DT[,(input_col_names) := 0]
      
    }else{
      soln <- deSolve::ode(y = exprs[i, ], times = times, 
                           func = my_ode, parms = externalInputs[i, ])
      soln_DT <- data.table(soln)
      soln_DT[,sample:=i ]
      input_col_names <- paste(names(externalInputs[i, ]),"input", sep = "_")
      soln_DT[,(input_col_names) := as.list(externalInputs[i, ])]
    }
    
    return(soln_DT)
  })
    
  res = rbindlist(res)
  first_cols = c("sample","time", grep("_input",names(res), value = T))
  setcolorder(res, 
              neworder = c(first_cols, setdiff(names(res), first_cols)))
  
  
  #add experimental noise
  all_genes <- setdiff(names(res),c("sample","time"))
  res[,(all_genes) := lapply(.SD, addNormNoise, 
                             noise = simulation@expnoise),
      .SDcols = all_genes]
  
  
  return(list(emat = res))
}

