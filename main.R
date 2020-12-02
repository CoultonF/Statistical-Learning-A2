# Title     : STAT652-A1
# Objective : Minimize R2
# Created by: coultonf
# Created on: 2020-11-06
#Libraries for...
#Linear Classifier
library(FNN)
#Variable selection
library(dplyr)
library(leaps)
# log reg
library(glmnet)
# multinomial log linear model
library(nnet)
#svm
library(e1071)
#random forest
library(randomForest)
#lda
library(MASS)
#rpart
library(rpart)
#gbm
library(gbm)
#Seed default

seed = 10

#Helper functions
get.folds = function(n, K) {
  set.seed(seed)
  n.fold = ceiling(n / K)
  fold.ids.raw = rep(1:K, times = n.fold)
  fold.ids = fold.ids.raw[1:n]
  folds.rand = fold.ids[sample.int(n)]
  return(folds.rand)
}
get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}
shuffle <- function(x, seed=1){
  set.seed(seed)
  new_order = sample.int(length(x))
  new_x = x[new_order]
  return(new_x)
}
predict.matrix = function(fit.lm, X.mat){
  coeffs = fit.lm$coefficients
  Y.hat = X.mat %*% coeffs
  return(Y.hat)
}
misclassify.rate = function(Y, Y.hat){
  return(mean(Y != Y.hat))
}
#Variable Selection Functions
leaps.selection = function(data){
  allsub = regsubsets(x=data[,-1], y=data[,1], nbest=1,nvmax=15)
  summ.1 = summary(allsub)
  summ.1$bic
  par(mfrow=c(1,1))
  plot(allsub, main="All Subsets on half of data data")
  n1 = nrow(data)
  results1 = matrix(data=NA, nrow=16, ncol=4)
  mod0 = lm(Y ~ 1, data=data)
  pred11 = predict(mod0, newdata=data)
  sMSE = mean((pred11-data$Y)^2)
  BIC = extractAIC(mod0, k=log(n1))
  MSPE = mean((pred2-data[data$set==2,]$Y)^2)
  results1[1,] = c(0, sMSE, BIC[2], MSPE)
  colnames(results1) = c("p", "sMSE", "BIC", "MSPE")
  data2 = data[,c(1:15)]
  head(data)
  for(v in 1:15){
    mod1 = lm(Y ~ ., data=data[data$set==1, summ.1$which[v,]])
    BIC = extractAIC(mod1, k=log(n1))
    pred1 = predict(mod1)
    sMSE = mean((pred1-data[data$set==1,]$Y)^2)
    pred2 = predict(mod1, newdata=data[data$set==2,])
    MSPE = mean((pred2-data[data$set==2,]$Y)^2)
    results1[v+1,] = c(v, sMSE, BIC[2], MSPE)
  }
  round(results1, digits=2)
  # Best size according to BIC
  results1[which.min(results1[,3]),1]
  # Best size according to MSPE
  results1[which.min(results1[,4]),1]

  # All 3 plots together
  par(mfrow=c(1,3))
  plot(x=results1[,1], y=results1[,2], xlab="Vars in model", ylab="sample-MSE",
       main="SampleMSE vs Vars: 1st", type="b")
  plot(x=results1[,1], y=results1[,3], xlab="Vars in model", ylab="BIC",
       main="BIC vs Vars: 1st", type="b")
  plot(x=results1[,1], y=results1[,4], xlab="Vars in model", ylab="MSPE",
       main="MSPE vs Vars: 1st", type="b")
#  return best data
  return(data)
}

knn.classifier = function(X.train, X.valid, Y.train, Y.valid){
  K.max = 40 # Maximum number of neighbours

  ### Container to store CV misclassification rates
  mis.CV = rep(0, times = K.max)

  for(i in 1:K.max){
    ### Fit leave-one-out CV
    this.knn = knn.cv(X.train, Y.train, k=i)

    ### Get and store CV misclassification rate
    this.mis.CV = mean(this.knn != Y.train)
    mis.CV[i] = this.mis.CV
  }
  k.min = which.min(mis.CV)
  SE.mis.CV = sapply(mis.CV, function(r){
    sqrt(r*(1-r)/nrow(X.train))
  })
  thresh = mis.CV[k.min] + SE.mis.CV[k.min]
  k.1se = max(which(mis.CV <= thresh))
  knn.1se = knn(X.train, X.valid, Y.train, k.1se)
  return (misclassify.rate(knn.1se, Y.valid))
}

glm.classifier = function(X.train, X.valid, Y.train, Y.valid){
  logit.fit <- cv.glmnet(x=as.matrix(X.train),
                    y=Y.train, family="multinomial")
  lambda.min = logit.fit$lambda.min
  lambda.1se = logit.fit$lambda.1se
  Y.hat <- predict(logit.fit, type="class",
                             s=logit.fit$lambda.1se,
                             newx=as.matrix(X.valid))
  return(misclassify.rate(Y.valid, Y.hat))

}

lda.classifier = function(X.train, X.valid, Y.train, Y.valid){
  fit.lda = lda(X.train, Y.train)
  pred.class <- predict(fit.lda,X.valid)$class
  return(misclassify.rate(Y.valid, pred.class))
}

mll.classifier = function(X.train, X.valid, Y.train, Y.valid){
  fit.log.nnet = multinom(Y.train ~ ., data = cbind(X.train, Y.train))
  Y.hat = predict(fit.log.nnet, newdata=X.valid, type="class")
  table(Y.hat, Y.valid, dnn = c("<MLL> Predicted", "Observed"))
  return(misclassify.rate(Y.valid, Y.hat))

}
rf.classifier = function (X.train, X.valid, Y.train, Y.valid){
  this.fit.rf = randomForest(data=cbind(X.train, Y.train), Y.train~., mtry=5, nodesize=5,
                      importance=TRUE, keep.forest=TRUE, ntree=500)
  Y.hat = predict(this.fit.rf, newdata = X.valid)
  table(Y.hat, Y.valid, dnn = c("RF Predicted", "Observed"))
  return(misclassify.rate(Y.valid, Y.hat))
}

regtree.classifier = function(X.train, X.valid, Y.train, Y.valid){
  reg.tree = rpart(data=cbind(Y.train, X.train), method="class", Y.train ~ ., cp=0)
  cpt = reg.tree$cptable
  minrow <- which.min(cpt[,4])
  # Take geometric mean of cp values at min error and one step up
  cplow.min <- cpt[minrow,1]
  cpup.min <- ifelse(minrow==1, yes=1, no=cpt[minrow-1,1])
  cp.min <- sqrt(cplow.min*cpup.min)

  # Find smallest row where error is below +1SE
  se.row <- min(which(cpt[,4] < cpt[minrow,4]+cpt[minrow,5]))
  # Take geometric mean of cp values at min error and one step up
  cplow.1se <- cpt[se.row,1]
  cpup.1se <- ifelse(se.row==1, yes=1, no=cpt[se.row-1,1])
  cp.1se <- sqrt(cplow.1se*cpup.1se)
  reg.tree.cv.1se <- prune(reg.tree, cp=cp.1se)
  Y.hat <- predict(reg.tree.cv.1se, newdata=X.valid, type="class")
  return(misclassify.rate(Y.valid, Y.hat))

}
lognet.classifier = function(X.train, X.valid, Y.train, Y.valid){
  Y.train.num = class.ind(Y.train)
  MSE.best = Inf    ### Initialize sMSE to largest possible value (infinity)
  M = 20            ### Number of times to refit.

  for(i in 1:M){
    ### For convenience, we stop nnet() from printing information about
    ### the fitting process by setting trace = F.
    this.nnet = nnet(X.train, Y.train.num, size = 6, decay = 0.1, maxit = 2000,
      softmax = T, trace = F)
    this.MSE = this.nnet$value
    if(this.MSE < MSE.best){
      NNet.best.0 = this.nnet
      MSE.best = this.MSE
    }
  }
  ### Now we can evaluate the validation-set performance of our naive neural
  ### network. We can get the predicted class labels using the predict()
  ### function and setting type to "class"
  Y.hat = predict(NNet.best.0, X.valid, type = "class")
  return(misclassify.rate(Y.valid, Y.hat))
}

svm.classifier = function (X.train, X.valid, Y.train, Y.valid){
  fit.svm.0 = svm(Y.train ~ ., data = cbind(Y.train, X.train), kernel = "radial",
  cost = 10, gamma = 2, degree=5)
  Y.hat = predict(fit.svm.0, X.valid)
  return(misclassify.rate(Y.valid, Y.hat))
}

#Load Data
data = na.omit(read.csv("Statistical-Learning-A2/P2Data2020.csv"))
#data = data[,c(1,7+1,8+1,9+1,11+1,14+1)]
test = na.omit(read.csv("Statistical-Learning-A2/P2Data2020testX.csv"))
#test = test[,c(7,8,9,11,14)]
data$Y = factor(data$Y, labels=c("A", "B", "C", "D", "E"))

#tune.nn(data)
#leaps.selection(data)

#Get num rows as n
n = nrow(data)

#Split into CV folds
K=10
folds = get.folds(n, K)

#CV Comparison of Diff Models
all.models = c("KNN", "LDA", "MLL", "REG", "RF", "NN", "SVM", "GLM")
#all.models = c("REG")
all.misclassify.rate = array(0, dim = c(K,length(all.models)))
colnames(all.misclassify.rate) = all.models


set.seed (8646824, kind="Mersenne-Twister")
perm <- sample ( x = nrow ( data ))
set1 <- data [ which ( perm <= 3* nrow ( data )/4) , ]
set2 <- data [ which ( perm > 3* nrow ( data )/4) , ]
this.fit.rf = randomForest(Y ~ ., data = set1)
pred.test.rf = predict(this.fit.rf, newdata = set2[-1])
(misclass.test.cv.1se <- mean(ifelse(pred.test.rf == set2$Y, yes=0, no=1)))

importnce = importance(this.fit.rf)
importnce = cbind(c("Features",colnames(set1[-1])),rbind(colnames(importnce), importnce))
varImpPlot(this.fit.rf)


# CV method
for(i in 1:K){
  X.train = data[folds != i,-1]
  X.valid = data[folds == i,-1]
  Y.train = data[folds != i,1]
  Y.valid = data[folds == i,1]
  X.train.scaled = rescale(X.train, X.train)
  X.valid.scaled = rescale(X.valid, X.train)

  #KNN
  all.misclassify.rate[i, "KNN"] = knn.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #  LDA
  all.misclassify.rate[i, "LDA"] = lda.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #  multinom log linear net
  all.misclassify.rate[i, "MLL"] = mll.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #  multinom log linear net
  all.misclassify.rate[i, "REG"] = regtree.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #Random Forest
  all.misclassify.rate[i, "RF"] = rf.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #NNET
  all.misclassify.rate[i, "NN"] = lognet.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #SVM
  all.misclassify.rate[i, "SVM"] = svm.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

  #GLM
  all.misclassify.rate[i, "GLM"] = glm.classifier(X.train.scaled, X.valid.scaled, Y.train, Y.valid)

}
par(mfrow=c(1,1))
par(mar=c(5,4,4,2))
boxplot(all.misclassify.rate, main = paste0("CV Misclassifiers over ", K, " folds"))
#all.RMSPEs = apply(all.MSPEs, 2, function(W) W/min(W))
#boxplot(t(all.RMSPEs))
##Testing R2 on self
#rsq <- function(x, y) summary(lm(y~x))$r.squared
#
#Y.hat = gam.model(data[,-1], data[,1], test, test[,1])[[2]]
#Y = gam.model(data[,-1], data[,1], data[,-1], data[1])[[2]]
#summary(Y)
#summary(data[,1])
#summary(Y.hat)
#rsq(data[,1], Y)
#
#write.table(Y.hat, 'output.csv', sep = ",", row.names = F, col.names = F)

p.train = 0.75
n = nrow(data)
n.train = floor(p.train*n)

ind.random = sample(1:n)
data.train = data[ind.random <= n.train,]
X.train.raw = data.train[,-1]
Y.train = data.train[,1]
data.valid = data[ind.random > n.train,]
X.valid.raw = data.valid[,-1]
Y.valid = data.valid[,1]

X.train = rescale(X.train.raw, X.train.raw)
X.valid = rescale(X.valid.raw, X.train.raw)

#data = na.omit(read.csv("P2Data2020.csv"))
#test = na.omit(read.csv("P2Data2020testX.csv"))
#data$Y = factor(data$Y, labels=c("A", "B", "C", "D", "E"))
fit.rf = randomForest(data=cbind(X.train, Y.train), Y.train~., mtry=5, nodesize=5, importance=TRUE, keep.forest=TRUE, ntree=500)
misclassify.rate(Y.valid, Y.hat)
t = table(Y.valid, Y.hat, dnn = c("Obs", "Pred"))
Y.hat = predict(fit.rf, newdata = X.valid)
write.table(Y.hat, 'output.csv', sep=',', row.names = F, col.names = F)







max.trees = 10000
all.shrink = c(0.001, 0.01, 0.1)
all.depth = c(2, 3, 4, 5, 6)
all.pars = expand.grid(shrink = all.shrink, depth = all.depth)
n.pars = nrow(all.pars)

### Number of folds
K = 5
n = nrow(data)
### Create folds
folds = get.folds(n, K)

### Create container for CV MSPEs
CV.MSPEs2 = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))

  ### Split data
  data.train = data[folds != i,]
  data.valid = data[folds == i,]
  Y.valid = data.valid$Y



  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]

    ### Fit model using current parameter values.
    fit.gbm = gbm(Y ~ ., data = data.train, distribution = "multinomial",
      n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink,
      bag.fraction = 0.8)

    ### Choose how many trees to keep using Tom's rule. This will print many
    ### warnings about not just using the number of trees recommended by
    ### gbm.perf(). We have already addressed this problem though, so we can
    ### just ignore the warnings.
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2

    ### Check to make sure that Tom's rule doesn't tell us to use more than 1000
    ### trees. If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }

    ### Get predictions and MSPE, then store MSPE
    Y.hat = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = misclassify.rate(Y.hat, Y.valid)

    CV.MSPEs2[i, j] = MSPE.gbm # Be careful with indices for CV.MSPEs
  }
}