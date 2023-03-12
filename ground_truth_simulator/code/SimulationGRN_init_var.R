#----SimulationGRN----
setValidity('SimulationGRN', validSimulationGRN)

setMethod(
	f = 'initialize',
	signature = 'SimulationGRN',
	definition = initSimulationGRN
)

setMethod(
	f = 'show',
	signature = 'SimulationGRN',
	definition = function(object) {
		mxp = 10
		
		cat('Graph:', length(object@graph@nodeset), 'nodes,', length(object@graph@edgeset), 'edges', '\n')
		cat('Experimental noise sd:', object@expnoise, '\n')
		cat('Biological noise sd:', object@bionoise, '\n')
		cat('Randomization seed:', object@seed, '\n')
	}
)

setMethod(
	f = '$',
	signature = 'SimulationGRN',
	definition = function(x, name) {
	  value = slot(x, name)
	  
	  if (name %in% 'solution'){
	    soln = x@solution
	    soln = c(soln$x, x@externalInputs)
	    soln = soln[order(names(soln))]
	    value = soln
	  }
	  
		return(value)
	}
)

setMethod(
	f = '$<-',
	signature = 'SimulationGRN',
	definition = function(x, name, value) {
		slot(x, name)<-value
		validObject(x)
		return(x)
	}
)

#----SimulationGRN: generateInputModels----
setGeneric(
  name = 'generateInputModels',
  def = function(simulation, propBimodal) {
    standardGeneric('generateInputModels')
  }
)

setMethod(
  f = 'generateInputModels',
  signature = c('SimulationGRN'),
  definition = createInputModels
)

#----SimulationGRN: simulateDataset----
setGeneric(
  name = 'simulateDataset',
  def = function(simulation, numsamples, cor.strength, externalInputs, timeStamps, inputGeneVar, outputGeneVar) {
    standardGeneric('simulateDataset')
  }
)

setMethod(
  f = 'simulateDataset',
  signature = c('SimulationGRN', 'numeric', 'missing', 'missing','missing', 'numeric', 'numeric'),
  definition = simDataset
)

setMethod(
  f = 'simulateDataset',
  signature = c('SimulationGRN', 'numeric', 'numeric', 'missing','numeric', 'numeric', 'numeric'),
  definition = simDataset
)

setMethod(
  f = 'simulateDataset',
  signature = c('SimulationGRN', 'numeric', 'missing', 'ANY'),
  definition = simDataset
)



