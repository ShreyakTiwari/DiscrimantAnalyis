rm(list=ls())

parkinsons<-read.csv('/Users/shreyaktiwari/Downloads/coursework/machine learning/dataset/parkinsons.data')
iris<-read.csv('/Users/shreyaktiwari/Downloads/coursework/machine learning/dataset/iris.txt',header=FALSE)

#creating training and test data using sampling
set.seed(1204)

#for iris data set
train=sample(1:nrow(iris),2*nrow(iris)/3)
train_data = iris[train,]
test_data  = iris[-train,] 

Y<-train_data$V5
X_subset<-subset(train_data,select=-c(5))
labelCol<-5

#for parkinson dataset
train=sample(1:nrow(parkinsons),2*nrow(parkinsons)/3)
train_data = parkinsons[train,]
test_data  = parkinsons[-train,] 

Y<-train_data$status
X_subset<-subset(train_data,select=-c(name,status))
labelCol<-which(names(parkinsons)=='status')

#finding labels of dataset
label=sort(as.numeric(names(summary(as.factor(Y)))))

#checking best attributes
vector<-matrix(1:ncol(X_subset),1,ncol(X_subset))

max_index     = order(apply(vector,2,bestAttribute),decreasing = T)[1]
sec_max_index = order(apply(vector,2,bestAttribute),decreasing = T)[2]

#for best attribute
X<-as.matrix(X_subset[,c(max_index)])
Z<-subset(test_data,select=c(sec_max_index))

#for best two attributes
X<-X_subset[,c(max_index,sec_max_index)]
Z<-subset(test_data,select=c(max_index,sec_max_index))

#for all  attributes
X<-X_subset;

#for iris
Z<-subset(test_data,select=-c(5))

#for parkinson
Z<-subset(test_data,select=-c(name,status))

#for all attributes QDA , here we remove highly correlated columns 
tmp <- cor(X_subset)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
X <- X_subset[,apply(tmp,2,function(x) all(x <= 0.9))]
Z<-subset(test_data,select=names(X))

LDA(Z);
QDA(Z);

LDA <- function(Z){
  #value of pi as per K class ( 0 or 1)
  pi0=length(Y[Y==label[1]])/length(Y)
  pi1=length(Y[Y==label[2]])/length(Y)
  sigma1=c();
  sigma2=c();
  
  if(dim(Z)[2]==1){
    for(i in 1:dim(Z)[1]){
      #calculation of sigma
      sigma1 = append(sigma1,as.matrix(Z[i,])%*%solve(cov(X))%*%(mean(X[Y==label[1]]))-0.5*t(mean(X[Y==label[1]]))%*%solve(cov(X))%*%mean(X[Y==label[1]])+log(pi0))
      sigma2 = append(sigma2,as.matrix(Z[i,])%*%solve(cov(X))%*%(mean(X[Y==label[2]]))-0.5*t(mean(X[Y==label[2]]))%*%solve(cov(X))%*%mean(X[Y==label[2]])+log(pi1))
    }
  }
  else{
    for(i in 1:dim(Z)[1]){
      #calculation of sigma
      sigma1 = append(sigma1,as.matrix(Z[i,])%*%solve(cov(X))%*%(colMeans(X[Y==label[1],]))-0.5*t(colMeans(X[Y==label[1],]))%*%solve(cov(X))%*%colMeans(X[Y==label[1],])+log(pi0))
      sigma2 = append(sigma2,as.matrix(Z[i,])%*%solve(cov(X))%*%(colMeans(X[Y==label[2],]))-0.5*t(colMeans(X[Y==label[2],]))%*%solve(cov(X))%*%colMeans(X[Y==label[2],])+log(pi1))
    }
  }
  compare(sigma1,sigma2);
}

QDA <- function(Z){
  #value of pi as per K class ( 0 or 1)
  pi0=length(Y[Y==label[1]])/length(Y)
  pi1=length(Y[Y==label[2]])/length(Y)
  sigma1=c();
  sigma2=c();
  
  if(dim(Z)[2]==1){
    for(i in 1:dim(Z)[1]){
      #calculation of sigma
      sigma1 = append(sigma1,-0.5*log(det(cov(as.matrix(X[Y==label[1]]))))-0.5*as.matrix(Z[i,])%*%solve(cov(as.matrix(X[Y==label[1]])))%*%t(as.matrix(Z[i,]))+as.matrix(Z[i,])%*%solve(cov(as.matrix(X[Y==label[1]])))%*%colMeans(as.matrix(X[Y==label[1]]))-0.5%*%t(colMeans(as.matrix(X[Y==label[1]])))%*%solve(cov(as.matrix(X[Y==label[1]])))%*%colMeans(as.matrix(X[Y==label[1]]))+log(pi0))
      sigma2 = append(sigma2,-0.5*log(det(cov(as.matrix(X[Y==label[2]]))))-0.5*as.matrix(Z[i,])%*%solve(cov(as.matrix(X[Y==label[2]])))%*%t(as.matrix(Z[i,]))+as.matrix(Z[i,])%*%solve(cov(as.matrix(X[Y==label[2]])))%*%colMeans(as.matrix(X[Y==label[2]]))-0.5%*%t(colMeans(as.matrix(X[Y==label[2]])))%*%solve(cov(as.matrix(X[Y==label[2]])))%*%colMeans(as.matrix(X[Y==label[2]]))+log(pi1))
    }
  }
  else{
    for(i in 1:dim(Z)[1]){
      #calculation of sigma
      sigma1 = append(sigma1,-0.5*log(det(cov(X[Y==label[1],])))-0.5*as.matrix(Z[i,])%*%solve(cov(X[Y==label[1],]))%*%t(as.matrix(Z[i,]))+as.matrix(Z[i,])%*%solve(cov(X[Y==label[1],]))%*%colMeans(X[Y==label[1],])-0.5%*%t(colMeans(X[Y==label[1],]))%*%solve(cov(X[Y==label[1],]))%*%colMeans(X[Y==label[1],])+log(pi0))
      sigma2 = append(sigma2,-0.5*log(det(cov(X[Y==label[2],])))-0.5*as.matrix(Z[i,])%*%solve(cov(X[Y==label[2],]))%*%t(as.matrix(Z[i,]))+as.matrix(Z[i,])%*%solve(cov(X[Y==label[2],]))%*%colMeans(X[Y==label[2],])-0.5%*%t(colMeans(X[Y==label[2],]))%*%solve(cov(X[Y==label[2],]))%*%colMeans(X[Y==label[2],])+log(pi1))
    }
  }
  compare(sigma1,sigma2);
}

#comparing value of sigma of two diff classes and assignaing label for which class
#sigma is greatest
compare <- function(sigma1,sigma2){
  Z$status<-0
  Z$status[which(as.data.frame(sigma1)>as.data.frame(sigma2))]<-label[1]
  Z$status[which(as.data.frame(sigma1)<as.data.frame(sigma2))]<-label[2]
  bool <- Z$status==test_data[,labelCol]
  print(length(bool[bool==TRUE])/length(Z$status))
  table(Z$status,test_data[,labelCol])
}

bestAttribute<-function(i){
  mean.0 <- mean(X_subset[(Y==label[1]),i])
  mean.1 <- mean(X_subset[(Y==label[2]),i])
  st.dev <- sd(X_subset[,i])
  return ((mean.1-mean.0)/st.dev)
}