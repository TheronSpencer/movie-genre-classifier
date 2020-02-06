setwd("~/MATH445")

#load necessary libraries
library(tm) 
library(fastNaiveBayes)
library(dplyr)
library(Matrix)
library(caret)  # Load caret package
library(randomForest)
library(rpart)
library(caTools)
library(nnet)

  #read in data
  movies <- read.csv("cleanedMovies1.csv", header = TRUE)
  movies <- data.frame(movies)
  movies$PlotClean <- as.character(movies$PlotClean)
  movies$GenreSplitMain <- as.character(movies$GenreSplitMain)
  movies <- movies[,2:3] #only need plot and genre
  glimpse(movies)
  
  sort(table(movies$GenreSplitMain))
  
  #we'll only focus on the following 4 genres as they are the most prevalent 
  movies <- subset(movies, movies$GenreSplitMain == "['drama']" | movies$GenreSplitMain == "['comedy']" |
                            movies$GenreSplitMain == "['action']" | movies$GenreSplitMain == "['thriller']" )
           
  movies$GenreSplitMain <- as.factor(movies$GenreSplitMain)
  sort(table(movies$GenreSplitMain))
  
  dim(movies) #17134 movies totoal
  
  #randomize
  movies <- movies[sample(dim(movies)[1]), ]
  movies <- movies[sample(dim(movies)[1]), ]
  
  #turn movie plots into corpus
  corpus <- Corpus(VectorSource(movies$PlotClean))
  corpus

  #clean corpus by removing puncuation, stop words, etc
  #ignore warning messages
  cleanCorpus <- tm_map(corpus, content_transformer(tolower))
  cleanCorpus <- tm_map(cleanCorpus, removeNumbers) 
  cleanCorpus <- tm_map(cleanCorpus, removePunctuation) 
  cleanCorpus <- tm_map(cleanCorpus, removeWords, stopwords(kind="SMART"))
  cleanCorpus <- tm_map(cleanCorpus, stripWhitespace)

  #corpus is no longer needed
  rm(corpus)
  
  #create documentTermMatrix
  dtm <- DocumentTermMatrix(cleanCorpus)
  
  ?removeSparseTerms
  #remove sparse words
  sparse <- removeSparseTerms(dtm, 0.93)
  dim(sparse)
  
  #create dtm of word count as well as binary instance dtm where 
  # for the binary, 1 = word is present, 0 = word is absent
  dtm_count <- as.data.frame(as.matrix(sparse))
  dtm_binary <- as.data.frame(as.matrix((dtm_count > 0)+0))
  
  # view segments of matrices to get an idea
  # dtm_count[1:10,1:10]
  # dtm_binary[1:10,1:10]
  # inspect(dtm[40:50, 10:15])
  
  # split data into training (70%) and testing sets (30%)
  sevenTenths <- round(dim(movies)[1]*0.7)
  movies.train <- movies[1:sevenTenths,]
  movies.test <- movies[(sevenTenths+1):dim(movies)[1],]

  dim(movies.test)
  dim(dtm)
  
  dtm_count.train <- dtm_count[1:sevenTenths,]
  dtm_count.test <- dtm_count[(sevenTenths+1):dim(movies)[1],]
  
  dtm_binary.train <- dtm_binary[1:sevenTenths,]
  dtm_binary.test <- dtm_binary[(sevenTenths+1):dim(movies)[1],]
  
 
#Naive bayes  

  #for binary instance dtm
  NBcount <- fastNaiveBayes.multinomial(as.matrix(dtm_binary.train), as.factor(movies.train$GenreSplitMain), laplace = 1)
  NBcount.predict <- predict(NBcount, as.matrix(dtm_binary.test))
  conf.mat <- confusionMatrix(NBcount.predict, movies.test$GenreSplitMain)
  round(conf.mat$overall, digits = 5)
  
  #for word frequency dtm
  NBbinary.predict = predict(fastNaiveBayes.multinomial(as.matrix(dtm_count.train), as.factor(movies.train$GenreSplitMain), laplace = 1), dtm_count.test)
  conf.mat <- confusionMatrix(NBbinary.predict, movies.test$GenreSplitMain)
  round(conf.mat$overall, digits = 5)
  

#Random forest

  rf <- randomForest(x = as.matrix(dtm_binary.train), y = movies.train$GenreSplitMain)
  
  pred <- predict(rf, as.matrix(dtm_binary.test))
  round(confusionMatrix(pred, movies.test$GenreSplitMain)$overall, digits = 5)
  
#Multinomial logistic regression
  # put training/ testing set into matrices that contain the features and 
  # respected classes so data can be fed into multinomial model
  training <- as.data.frame(as.matrix(dtm_binary.train))
  training <- cbind(movies.train, training)
  testing <- as.data.frame(as.matrix(dtm_binary.test))
  testing <- cbind(movies.test, testing)
  training$`movies.train$GenreSplitMain` <- NULL
  testing$`movies.test$GenreSplitMain` <- NULL
  spl <- sample.split(movies$GenreSplitMain, .7)
  
  final_training <- training[spl == T,]
  final_testing <- testing[spl == F,]
  glimpse(final_testing)
  dim(final_training)
  
  #relevel, set action genre as base
  final_training$GenreSplitMain <- relevel(final_training$GenreSplitMain, ref = "['action']")
  
  #create model
  log.model <- multinom(GenreSplitMain~., 
                        data = final_training[1:dim(final_training)[1],2:dim(final_training)[2]]
                        , MaxNWts = 15000)
  
  summary(log_model)  
  
  #evaluate performance on testing set
  log.pred <- log_model %>% predict(final_testing[1:dim(final_testing)[1],2:dim(final_testing)[2]])

  round(confusionMatrix(log.pred, final_testing$GenreSplitMain[1:dim(final_testing)[1]])$overall, digits = 4)
  
  #check residuals are normally distributed and constant variance assumption met
  library(nortest)
  ad.test(log.model$residuals)
  
  library(MASS)
  e.star = log_model$residuals
  y.hat=predict(log_model)
  plot(e.star~y.hat, ylim=c(-2,2), ylab="Residuals", 
       xlab="Treatment Mean", main="Plot of Residuals vs. Treatment Means")
  abline(h=2, col="blue", lty=2)
  abline(h=-2, col="blue", lty=2)
  abline(h=0)