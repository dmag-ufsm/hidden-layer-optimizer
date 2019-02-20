
# para instalar dependências descomente a linha abaixo
#install.packages("doParallel") 

# Carrega biblitecas necessárias
library(parallel)
library(doParallel)
library(neuralnet)

#Leio um csv. Use setwd() para definir o diretório atual
foraminiferos_TMA <- read.csv("foraminiferos_TMA.csv");

# Random sampling 70-30 ... Vamos definir essa distribuição como padrão
amostra = 0.70 * nrow(foraminiferos_TMA)

# Semente
set.seed(99)

# Amostras ....
indicie = sample( seq_len ( nrow ( foraminiferos_TMA ) ), size = amostra )

# Geralmente normalizamos os dados pela sigmoid, mas talvez nao seja necessário para essa escala.
# Caso contrário, usamos min-max 0-1
foraminiferos_TMA.norm <- foraminiferos_TMA

# Criar conjuntos de treino e teste
treino = foraminiferos_TMA.norm [indicie , ]
teste = foraminiferos_TMA.norm [-indicie , ]


# Gerar a formula. Temp_mediaAnual é o atributo alvo, todos os demais são passados após ~
# collapse é o separador. A fórmula tem que ser separada por '+'
vars <- names(treino)
formula <- as.formula(paste("Temp_mediaAnual ~", paste(vars[vars != "Temp_mediaAnual"], collapse = " + ")))

# Limites da base de testes

# limite inferior da primeira camada
l1_l <- 22
# limite superior da primeira camada
l1_h <- 25
# limite inferior da segunda camada
l2_l <- 10
# limite superior da segunda camada
l2_h <- 14


# O tamanho da base é calculado
tamanho_base <- (l1_h-l1_l+1)*(l2_h-l2_l+1)

# É criada a base: uma matriz contendo todos os possíveis combinações de tamanhos de camadas
base <- matrix(nrow = tamanho_base, ncol = 2)
k <- 1
for (i in l1_l:l1_h) {
  for (j in l2_l:l2_h) {
    base[k,1] <- i
    base[k,2] <- j
    k <- k + 1
  }
}

# Está é a função que realmente calcula o erro médio quadrático para uma determinada 
# combinação de tamanhos de camadas
work = function(l1,l2) {
  set.seed(9)
  
  # Criando a rede neural para os valores atuais
  NN = neuralnet(formula, treino, hidden = c(l1,l2), stepmax = 1e+06, linear.output = T)
  
  # Prevendo os valores
  previsao <- compute(NN,teste[,-45])
  
  # Calculando o erro médio quadrático (MSE)
  mse <- sum((teste$Temp_mediaAnual - previsao$net.result)^2)/nrow(teste)
  
  # Retorna uma entrada para o resultado final com os tamanhos das camadas e o MSE desta execução
  return(c(l1,l2,mse))
}


# Numero de processadores lógicos
no_cores <- max(1, detectCores() - 1)

# Cria o cluster que permite processamento paralelo
cl <- makeCluster(no_cores)

# Registra doParallel (%dopar%) no cluster criado
registerDoParallel(cl)

# Executa paralelamente a função definida como work para cada item da base
result <- foreach(index = 1:tamanho_base,
             # Método utilizado para combinar resultados
             .combine = rbind, 
             # Define quais variavéis poderão ser acessadas de dentro das threads
             .export = c("base","formula","treino","teste"), 
             # Define quais bibliotecas estarão disponiveis de dentro das threads
             .packages = c("neuralnet") 
             ) %dopar% work(base[index,1],base[index,2])


# Libera recursos de volta para o SO
stopCluster(cl)

# Define nomes para as colunas do resultado
colnames(result) <- c("1st Layer","2nd Layer","MSE")

# Salva o valor do resultado como csv sem a coluna de nomes de linhas
write.csv(result, file = "result.csv", row.names = FALSE)
