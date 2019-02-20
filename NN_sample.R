#Leio um csv. Use setwd() para definir o diret?rio atual
foraminiferos_TMA <- read.csv("foraminiferos_TMA.csv");

# Random sampling 70-30 ... Vamos definir essa distribui??o como padr?o
amostra = 0.70 * nrow(foraminiferos_TMA)
# semente
set.seed(99)

# amostras ....
index = sample( seq_len ( nrow ( foraminiferos_TMA ) ), size = amostra )
# Geralmente normalizamos os dados pela sigmoid, mas talvez nao seja necess?rio para essa escala.
# caso contr?rio, usamos min-max 0-1
foraminiferos_TMA.norm <- foraminiferos_TMA

# load library
library(neuralnet)
library(parallel)

# creating training and test set
treino = foraminiferos_TMA.norm [index , ]
teste = foraminiferos_TMA.norm [-index , ]

# Get a formula. Temp_mediaAnual ? o atributo alvo, todos os demais s?o passados ap?s ~
# collapse ? o separador. A f?rmula tem que ser separada por '+'
vars <- names(treino)
f <- as.formula(paste("Temp_mediaAnual ~", paste(vars[vars != "Temp_mediaAnual"], collapse = " + ")))

# fitting a neural network
# linear.output: T=regressao, F=Classificacao
# Aqui exploramos a rede neural. O par?metro hidden ? nosso alvo.
# stepmax indica quantas itera??es s?o permitidas para uma converg?ncia.
#
# Para saber mais usem: ??neuralnet
# set.seed(9)
# NN = neuralnet(f, treino, hidden = c(22,12) ,
#                stepmax = 1e+06, linear.output = T)


# plot(NN)

# Predicting the values
# previsao <- compute(NN,teste[,-45])


# Mean Squared Error. Queremos minimizar isso!!
# NN.MSE <- sum((teste$Temp_mediaAnual - previsao$net.result)^2)/nrow(teste)
# NN.MSE

min_i <- 23
max_i <- 23
min_j <- 15
max_j <- 15
min_k <- 0
max_k <- 0
extra_without_3rd_layer <- F

# Best Results so far:
# i <- 23
# y <- 15
# k <- 0

result <- array(dim = c(max_i-min_i+1,max_j-min_j+1,max_k-min_k+1+extra_without_3rd_layer,4)) 
for (i in min_i:max_i) {
  for (j in min_j:max_j) {
    for (k in min_k:(max_k+extra_without_3rd_layer)) {
      set.seed(9)
      
      if (k <= max_k && k > 0) {
        k_val = k
        NN = neuralnet(f, treino, hidden = c(i,j,k), stepmax = 1e+06, linear.output = T)
      } else { # without thrid layer
        k_val = -1
        NN = neuralnet(f, treino, hidden = c(i,j), stepmax = 1e+06, linear.output = T)
      }
      
      previsao <- compute(NN,teste[,-45])
      mse <- sum((teste$Temp_mediaAnual - previsao$net.result)^2)/nrow(teste)
      result[i-min_i+1,j-min_j+1,k-min_k+1,1] <- i
      result[i-min_i+1,j-min_j+1,k-min_k+1,2] <- j
      result[i-min_i+1,j-min_j+1,k-min_k+1,3] <- k_val
      result[i-min_i+1,j-min_j+1,k-min_k+1,4] <- mse
    }
  }
}

result <- matrix(result, ncol=4)

colnames(result) <- c("1st Layer","2nd Layer","3rd Layer","MSE")
result
write.csv(result, file = "result.csv", row.names = FALSE)

