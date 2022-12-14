library(bnlearn)
library(randomForest)
set.seed(990806)
# rm(list = ls())

gs_m = function(data, var.missing, threshold = 0.1) {
  # find the intrinsic MB
  # inputs:
  # data: dataset with missing values
  # var.missing: names of missing variables
  # threshold: threshold for p-value
  # return:
  # mb_o: learned intrinsic MB
  mb_o = vector(mode = 'list', length = length(var.missing))
  names(mb_o) = var.missing
  for (s in var.missing) {
    candidate = rep(0, ncol(data) - 1)
    names(candidate) = seq(1, ncol(data))[-s]
    # forward
    while (TRUE) {
      for (can in names(candidate)) {
        idx = which(rowSums(is.na(data[, c(s, strtoi(can), mb_o[[paste(s)]])])) == 0)
        if (length(idx) != 0) {
          if (length(mb_o[[paste(s)]]) == 0) {
            candidate[can] = ci.test(colnames(data)[s], colnames(data)[strtoi(can)], data = data)$p.value
          } else {
            candidate[can] = ci.test(colnames(data)[s], colnames(data)[strtoi(can)], colnames(data)[mb_o[[paste(s)]]], data = data)$p.value
          } 
        }
      }
      if (length(candidate[candidate < threshold]) == 0) {
        break
      } else {
        mb_o[[paste(s)]] = c(mb_o[[paste(s)]], strtoi(names(which(candidate==min(candidate)))[1]))
        candidate = candidate[names(candidate) != names(which(candidate==min(candidate)))[1]]
      }
    }
    # backward
    candidate = mb_o[[paste(s)]][-length(mb_o[[paste(s)]])]
    for (can in candidate) {
      idx = which(rowSums(is.na(data[, c(s, strtoi(can), mb_o[[paste(s)]])])) == 0)
      if (length(idx) != 0) {
        p.value = ci.test(colnames(data)[s], colnames(data)[strtoi(can)], colnames(data)[setdiff(mb_o[[paste(s)]], can)], data = data)$p.value 
      } else {
        p.value = 0
      }
      if (p.value >= threshold) {
        mb_o[[paste(s)]] = setdiff(mb_o[[paste(s)]], can)
      }
    }
  }
  return(mb_o)
}

find.causes = function(data, var.missing, varType, threshold = 0.1) {
  # find the causes of missingness indicators
  # inputs:
  # data: dataset with missing values
  # test: CI test
  # return:
  # causes: the causes of missingness indicators
  data = droplevels(data)
  causes = vector(mode = 'list', length = length(var.missing))
  names(causes) = var.missing
  for (var in var.missing) {
    if (varType[var] == 'factor')
      data.missing = as.data.frame(as.factor(is.na(data[[var]])))
    else if (varType[var] == 'numeric')
      data.missing = as.numeric(is.na(data[[var]]))
    else
      stop('Not support mixed continuous and discrete data')
    data.missing = data.frame(data.missing)
    colnames(data.missing) = 'missing'
    causes[[paste(var)]] = seq(1, ncol(data))[-var]
    l = 0
    while (length(causes[[paste(var)]]) > l) {
      remaining.causes = causes[[paste(var)]]
      for (can in remaining.causes) {
        if (length(unique(data.missing[complete.cases(data[, can]), ])) == 2) {
          comb = combn(setdiff(remaining.causes, can), l)
          for (con_id in 1 : ncol(comb)) {
            data.temp = cbind(data.missing, data[, can], data[comb[, con_id]])
            data.temp = data.temp[complete.cases(data.temp), ]
            if (length(unique(data.temp$missing)) == 2) {
              if (ncol(data.temp) == 2) {
                p.value = ci.test(data.temp[, 1], data.temp[, 2])$p.value
              } else {
                p.value = ci.test(data.temp[, 1], data.temp[, 2], data.temp[, 3:ncol(data.temp)])$p.value
              }
              if (p.value > threshold) {
                causes[[paste(var)]] = setdiff(causes[[paste(var)]], can)
                break
              } 
            }
          } 
        }
      }
      l = l + 1
      if (l > 5) {
        break
      }
    }
  }
  return(causes)
}

missForest <- function(xmis, maxiter = 10, ntree = 100, variablewise = FALSE,
                       decreasing = FALSE, verbose = FALSE, replace = TRUE,
                       classwt = NULL, cutoff = NULL, strata = NULL,
                       sampsize = NULL, nodesize = NULL, maxnodes = NULL,
                       xtrue = NA, fs = "mbfs", threshold=0.1)
{ ## ----------------------------------------------------------------------
  ## Arguments:
  ## xmis         = data matrix with missing values
  ## maxiter      = stop after how many iterations (default = 10)
  ## ntree        = how many trees are grown in the forest (default = 100)
  ## variablewise = (boolean) return OOB errors for each variable separately
  ## decreasing   = (boolean) if TRUE the columns are sorted with decreasing
  ##                amount of missing values
  ## verbose      = (boolean) if TRUE then missForest returns error estimates,
  ##                runtime and if available true error during iterations
  ## mtry         = how many variables should be tried randomly at each node
  ## replace      = (boolean) if TRUE bootstrap sampling (with replacements)
  ##                is performed, else subsampling (without replacements)
  ## classwt      = list of priors of the classes in the categorical variables
  ## cutoff       = list of class cutoffs for each categorical variable
  ## strata       = list of (factor) variables used for stratified sampling
  ## sampsize     = list of size(s) of sample to draw
  ## nodesize     = minimum size of terminal nodes, vector of length 2, with
  ##                number for continuous variables in the first entry and
  ##                number for categorical variables in the second entry
  ## maxnodes     = maximum number of terminal nodes for individual trees
  ## xtrue        = complete data matrix
  ## fs           = feature selection method, if "mbfs" then call MF+MBFS 
  ##                algorithm, if "None" then call normal MF algorithm.
  ## threshold    = threshold for p value used in MBFS
  ##
  ## ----------------------------------------------------------------------
  ## Author: Daniel Stekhoven, stekhoven@nexus.ethz.ch
  
  ## stop in case of wrong inputs passed to randomForest
  n <- nrow(xmis)
  p <- ncol(xmis)
  if (!is.null(classwt))
    stopifnot(length(classwt) == p, typeof(classwt) == 'list')
  if (!is.null(cutoff))
    stopifnot(length(cutoff) == p, typeof(cutoff) == 'list')
  if (!is.null(strata))
    stopifnot(length(strata) == p, typeof(strata) == 'list')
  if (!is.null(nodesize))
    stopifnot(length(nodesize) == 2)
  
  ## remove completely missing variables
  if (any(apply(is.na(xmis), 2, sum) == n)){
    indCmis <- which(apply(is.na(xmis), 2, sum) == n)
    xmis <- xmis[,-indCmis]
    p <- ncol(xmis)
    cat('  removed variable(s)', indCmis,
        'due to the missingness of all entries\n')
  } 
  
  ## perform initial S.W.A.G. on xmis (mean imputation)
  ximp <- xmis
  varType <- character(p)
  for (t.co in 1:p) {
    if (is.numeric(xmis[[t.co]])) {
      varType[t.co] <- 'numeric'
      ximp[is.na(xmis[,t.co]),t.co] <- mean(xmis[[t.co]], na.rm = TRUE)
      next()
    } 
    if (is.factor(xmis[[t.co]])) {
      varType[t.co] <- 'factor'
      ## take the level which is more 'likely' (majority vote)
      max.level <- max(table(ximp[[t.co]]))
      ## if there are several classes which are major, sample one at random
      class.assign <- sample(names(which(max.level == summary(ximp[[t.co]]))), 1)
      ## it shouldn't be the NA class
      if (class.assign != "NA's") {
        ximp[is.na(xmis[[t.co]]),t.co] <- class.assign
      } else {
        while (class.assign == "NA's") {
          class.assign <- sample(names(which(max.level ==
                                               summary(ximp[[t.co]]))), 1)
        }
        ximp[is.na(xmis[[t.co]]),t.co] <- class.assign
      }
      next()
    }
    stop(sprintf('column %s must be factor or numeric, is %s', names(xmis)[t.co], class(xmis[[t.co]])))
  }
  
  ## extract missingness pattern
  NAloc <- is.na(xmis)            # where are missings
  noNAvar <- apply(NAloc, 2, sum) # how many are missing in the vars
  sort.j <- order(noNAvar)        # indices of increasing amount of NA in vars
  if (decreasing)
    sort.j <- rev(sort.j)
  sort.noNAvar <- noNAvar[sort.j]
  
  ## output
  Ximp <- vector('list', maxiter)
  
  ## initialize parameters of interest
  iter <- 0
  k <- length(unique(varType))
  convNew <- rep(0, k)
  convOld <- rep(Inf, k)
  OOBerror <- numeric(p)
  names(OOBerror) <- varType
  
  ## setup convergence variables w.r.t. variable types
  if (k == 1){
    if (unique(varType) == 'numeric'){
      names(convNew) <- c('numeric')
    } else {
      names(convNew) <- c('factor')
    }
    convergence <- c()
    OOBerr <- numeric(1)
  } else {
    names(convNew) <- c('numeric', 'factor')
    convergence <- matrix(NA, ncol = 2)
    OOBerr <- numeric(2)
  }
  
  ## function to yield the stopping criterion in the following 'while' loop
  stopCriterion <- function(varType, convNew, convOld, iter, maxiter){
    k <- length(unique(varType))
    if (k == 1){
      (convNew < convOld) & (iter < maxiter)
    } else {
      ((convNew[1] < convOld[1]) | (convNew[2] < convOld[2])) & (iter < maxiter)
    }
  }
  
  # feature selection by MBFS
  if ((fs == 'mbfs')) {
    var.missing = unname(which(noNAvar > 0))
    mb_o = gs_m(xmis, var.missing, threshold)
    cause.list = find.causes(xmis, var.missing, varType, threshold)
  }
  
  ## iterate missForest
  while (stopCriterion(varType, convNew, convOld, iter, maxiter)){
    if (iter != 0){
      convOld <- convNew
      OOBerrOld <- OOBerr
    }
    if (verbose){
      cat("  missForest iteration", iter+1, "in progress...")
    }
    t.start <- proc.time()
    ximp.old <- ximp
    
    for (s in 1 : p) {
      varInd <- sort.j[s]
      if (noNAvar[[varInd]] != 0) {
        if (fs == 'None') {
          obsi <- !NAloc[, varInd]
          misi <- NAloc[, varInd]
          obsY <- ximp[obsi, varInd]
          obsX <- ximp[obsi, seq(1, p)[-varInd]]
          misX <- ximp[misi, seq(1, p)[-varInd]]
          typeY <- varType[varInd]
          if (typeY == "numeric") {
            RF <- randomForest( x = obsX,
                                y = obsY,
                                ntree = ntree,
                                mtry = floor(sqrt(ncol(obsX) + 1)),
                                replace = replace,
                                sampsize = if (!is.null(sampsize)) sampsize[[varInd]] else
                                  if (replace) nrow(obsX) else ceiling(0.632 * nrow(obsX)),
                                nodesize = if (!is.null(nodesize)) nodesize[1] else 1,
                                maxnodes = if (!is.null(maxnodes)) maxnodes else NULL)
            ## record out-of-bag error
            OOBerror[varInd] <- RF$mse[ntree]
            misY <- predict(RF, misX)
          } else {
            obsY <- factor(obsY)
            summarY <- summary(obsY)
            if (length(summarY) == 1) {
              misY <- factor(rep(names(summarY), sum(misi)))
            } else {
              RF <- randomForest(x = obsX, 
                                 y = obsY, 
                                 ntree = ntree, 
                                 mtry = floor(sqrt(ncol(obsX) + 1)), 
                                 replace = replace, 
                                 classwt = if (!is.null(classwt)) classwt[[varInd]] else 
                                   rep(1, nlevels(obsY)),
                                 cutoff = if (!is.null(cutoff)) cutoff[[varInd]] else 
                                   rep(1 / nlevels(obsY), nlevels(obsY)),
                                 strata = if (!is.null(strata)) strata[[varInd]] else obsY, 
                                 sampsize = if (!is.null(sampsize)) sampsize[[varInd]] else 
                                   if (replace) nrow(obsX) else ceiling(0.632 * nrow(obsX)), 
                                 nodesize = if (!is.null(nodesize)) nodesize[2] else 5, 
                                 maxnodes = if (!is.null(maxnodes)) maxnodes else NULL)
              ## record out-of-bag error
              OOBerror[varInd] <- RF$err.rate[[ntree, 1]]
              ## predict missing parts of Y
              misY <- predict(RF, misX)
            }
          }
          ximp[misi, varInd] <- misY
        } else if (fs == 'mbfs') {
          obsi <- !NAloc[, varInd]
          misi <- NAloc[, varInd]
          obsY <- ximp[obsi, varInd]
          mb_X_v = mb_o[[paste(varInd)]]
          mb_X_r = c()
          for (v in var.missing) {
            if (varInd %in% cause.list[[paste(v)]]) {
              mb_X_r = c(mb_X_r, v)
              mb_X_v = unique(c(mb_X_v, setdiff(cause.list[[paste(v)]], varInd)))
            }
          }
          typeY <- varType[varInd]
          if (typeY == "numeric") {
            if (length(c(mb_X_v, mb_X_r)) != 0) {
              if (length(mb_X_r) == 0) {
                obsX <- ximp[obsi, mb_X_v]
                misX <- ximp[misi, mb_X_v]
              } else {
                obsX <- cbind(ximp[obsi, mb_X_v], matrix(NAloc[obsi, mb_X_r] * 1, nrow = sum(obsi)))
                colnames(obsX) = c(colnames(xmis)[mb_X_v], paste0(colnames(xmis)[mb_X_r], '_r'))
                misX <- cbind(ximp[misi, mb_X_v], matrix(NAloc[misi, mb_X_r] * 1, nrow = sum(misi)))
                colnames(misX) = c(colnames(xmis)[mb_X_v], paste0(colnames(xmis)[mb_X_r], '_r'))
              } 
            } else {
              obsX <- ximp[obsi, seq(1, p)[-varInd]]
              misX <- ximp[misi, seq(1, p)[-varInd]]
            }
            if (is.null(ncol(obsX))) {
              obsX = data.frame(obsX)
              colnames(obsX) = 'independent'
              misX = data.frame(misX)
              colnames(misX) = 'independent'
            }
            RF <- randomForest( x = obsX,
                                y = obsY,
                                ntree = ntree,
                                mtry = floor(sqrt(ncol(obsX) + 1)),
                                replace = replace,
                                sampsize = if (!is.null(sampsize)) sampsize[[varInd]] else
                                  if (replace) nrow(obsX) else ceiling(0.632 * nrow(obsX)),
                                nodesize = if (!is.null(nodesize)) nodesize[1] else 1,
                                maxnodes = if (!is.null(maxnodes)) maxnodes else NULL)
            ## record out-of-bag error
            OOBerror[varInd] <- RF$mse[ntree]
            misY <- predict(RF, misX)
          } else {
            if (length(mb_X_r) == 0) {
              obsX <- ximp[obsi, mb_X_v]
              misX <- ximp[misi, mb_X_v]
            } else {
              obsX <- cbind(ximp[obsi, mb_X_v], matrix(NAloc[obsi, mb_X_r], nrow = sum(obsi)))
              colnames(obsX) = c(colnames(xmis)[mb_X_v], paste0(colnames(xmis)[mb_X_r], '_r'))
              obs = data.frame(lapply(obsX, as.factor))
              misX <- cbind(ximp[misi, mb_X_v], matrix(NAloc[misi, mb_X_r], nrow = sum(misi)))
              colnames(misX) = c(colnames(xmis)[mb_X_v], paste0(colnames(xmis)[mb_X_r], '_r'))
              misX = data.frame(lapply(misX, as.factor))
              for (v in paste0(colnames(xmis)[mb_X_r], '_r')) {
                if (nlevels(obsX[[v]]) < nlevels(misX[[v]])) {
                  obsX = obsX[, !names(obsX) %in% c(v)]
                  misX = misX[, !names(misX) %in% c(v)]
                } else {
                  levels(misX[[v]]) = levels(obsX[[v]]) 
                }
              }
            }
            if (is.null(ncol(obsX))) {
              obsX = data.frame(obsX)
              colnames(obsX) = 'independent'
              misX = data.frame(misX)
              colnames(misX) = 'independent'
            }
            obsY <- factor(obsY)
            summarY <- summary(obsY)
            if (length(summarY) == 1) {
              misY <- factor(rep(names(summarY), sum(misi)))
            } else {
              RF <- randomForest(x = obsX, 
                                 y = obsY, 
                                 ntree = ntree, 
                                 mtry = floor(sqrt(ncol(obsX) + 1)), 
                                 replace = replace, 
                                 classwt = if (!is.null(classwt)) classwt[[varInd]] else 
                                   rep(1, nlevels(obsY)),
                                 cutoff = if (!is.null(cutoff)) cutoff[[varInd]] else 
                                   rep(1 / nlevels(obsY), nlevels(obsY)),
                                 strata = if (!is.null(strata)) strata[[varInd]] else obsY, 
                                 sampsize = if (!is.null(sampsize)) sampsize[[varInd]] else 
                                   if (replace) nrow(obsX) else ceiling(0.632 * nrow(obsX)), 
                                 nodesize = if (!is.null(nodesize)) nodesize[2] else 5, 
                                 maxnodes = if (!is.null(maxnodes)) maxnodes else NULL)
              ## record out-of-bag error
              OOBerror[varInd] <- RF$err.rate[[ntree, 1]]
              ## predict missing parts of Y
              misY <- predict(RF, misX)
            }
          }
          ximp[misi, varInd] <- misY
        } else {
          stop(paste('Unknown type of feature selection method:', fs))
        }
      }
    }
    if (verbose){
      cat('done!\n')
    }
    
    iter <- iter + 1
    Ximp[[iter]] <- ximp
    
    t.co2 <- 1
    ## check the difference between iteration steps
    for (t.type in names(convNew)){
      t.ind <- which(varType == t.type)
      if (t.type == 'numeric'){
        convNew[t.co2] <- sum((ximp[, t.ind] - ximp.old[, t.ind])^2) / sum(ximp[, t.ind]^2)
      } else {
        dist <- sum(as.character(as.matrix(ximp[, t.ind])) != as.character(as.matrix(ximp.old[, t.ind])))
        convNew[t.co2] <- dist / (n * sum(varType == 'factor'))
      }
      t.co2 <- t.co2 + 1
    }
    
    ## compute estimated imputation error
    if (!variablewise){
      NRMSE <- sqrt(mean(OOBerror[varType == 'numeric'])/
                      var(as.vector(as.matrix(xmis[, varType == 'numeric'])),
                          na.rm = TRUE))
      PFC <- mean(OOBerror[varType == 'factor'])
      if (k == 1){
        if (unique(varType) == 'numeric'){
          OOBerr <- NRMSE
          names(OOBerr) <- 'NRMSE'
        } else {
          OOBerr <- PFC
          names(OOBerr) <- 'PFC'
        }
      } else {
        OOBerr <- c(NRMSE, PFC)
        names(OOBerr) <- c('NRMSE', 'PFC')
      }
    } else {
      OOBerr <- OOBerror
      names(OOBerr)[varType == 'numeric'] <- 'MSE'
      names(OOBerr)[varType == 'factor'] <- 'PFC'
    }
    
    if (any(!is.na(xtrue))){
      err <- suppressWarnings(mixError(ximp, xmis, xtrue))
    }
    
    ## return status output, if desired
    if (verbose){
      delta.start <- proc.time() - t.start
      if (any(!is.na(xtrue))){
        cat("    error(s):", err, "\n")
      }
      cat("    estimated error(s):", OOBerr, "\n")
      cat("    difference(s):", convNew, "\n")
      cat("    time:", delta.start[3], "seconds\n\n")
    }
  }
  
  ## produce output w.r.t. stopping rule
  if (iter == maxiter){
    if (any(is.na(xtrue))){
      out <- list(ximp = Ximp[[iter]], OOBerror = OOBerr)
    } else {
      out <- list(ximp = Ximp[[iter]], OOBerror = OOBerr, error = err)
    }
  } else {
    if (any(is.na(xtrue))){
      out <- list(ximp = Ximp[[iter - 1]], OOBerror = OOBerrOld)
    } else {
      out <- list(ximp = Ximp[[iter - 1]], OOBerror = OOBerrOld,
                  error = suppressWarnings(mixError(Ximp[[iter - 1]], xmis, xtrue)))
    }
  }
  class(out) <- 'missForest'
  return(out)
}