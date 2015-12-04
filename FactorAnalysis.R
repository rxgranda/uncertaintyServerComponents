# Instalando las librerías que se utilizarán
install.packages("psych")
install.packages("nFactors")
install.packages("stats")

# Cargando las librerías que serán utilizadas
library(psych)
library(nFactors)
library(stats)

# dataset1.csv es el archivo utilizado originalmente en el paper de LAK14
# 333 estudiantes, todos graduados
data1 = read.table("dataset1.csv", sep=",", dec=".", h=T)

# dataset4.csv es el nuevo archivo generado por Aníbal
# 511 estudiantes de computación que tienen info para las materias
# analizadas
data2 = read.table("dataset4.csv", sep=",", dec=".", h=T)

# Qué dataset voy a utilizar ahorita?
grades = data1

# Del dataset seleccionado, tomo solo la info que corresponde a notas
# (esto porque en el dataset de Aníbal, hay más información aparte 
# de las notas.)
theGrades = grades[,2:27]

# Estandarizando los datos
theGrades <- as.data.frame(scale(theGrades))


# Generación del scree plot para observar, visualmente,
# cuál sería el número ideal de factores a extraer
# En este plot, el "codo" de la gráfica es siempre el mejor 
# candidato para esto.

ev <- eigen(cor(theGrades)) # get eigenvalues
ap <- parallel(subject=nrow(theGrades),var=ncol(theGrades),rep=100,cent=.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS)



# EJECUCIÓN DEL FACTOR ANÁLISIS COMO TAL #
# Se ejectuan tantos factor analysis como indique la variable 'totalFactors' 

totalFactors = 10

# Los resultados de cada factor analysis ejecutado se almacenan en una lista
# el elemento en la posición i de esta lista contiene los resultados
# del análisis cuando se extraen i factores
results = list()
for (i in 1:totalFactors) {
  results[[i]] = factanal(theGrades, i, rotation="varimax", scores = "regression")
}

# El arreglo pValues contitne los p-values de la prueba de hipótesis que indica
# si el número de factores extraídos es suficiente.
# El elemento en la posición i de este arreglo contiene el p-value calculado
# cuando se extraen i factores.
pValues = c()
for (i in 1:totalFactors) {
  pValues[i] = results[[i]]$PVAL
}

plot(pValues)



# La variable 'numbersOfFactos' sirve para seleccionar la solución
# (el análisis de factores) que deseemos inspeccionar a continuación
# Si 'numbersOfFactos' es x, la información generada a continuación
# corresponde al análisis en el que se extrajeron x factores

numbersOfFactos = 5

choosenSolution = results[[numbersOfFactos]]


# Este es el cutoff applicado a los valores de correlación
# para saber cuándo una variable es retenida en el factor o no
cuttOff = 0.3


# El código a continuación dibuja las materias contenidas en cada factor
# y almacena x archivos con la información de las mismas y el respectivo
# valor de correlación. Si se escogió x factores en la variable 'numbersOfFactos',
# entonces se generan x archivos (del factor1.csv al factorx.csv)

# Estos plots no tienen ningún significado estadístico; solo son usados para 
# verificar que el cutoff se haya aplicado correctamente

classes = list()

for (theFactor in 1:numbersOfFactos) {
  
  y = choosenSolution$loadings[,theFactor]
  x = 1:length(y)
  
  reducedY = y[y>=cuttOff]
  
  reducedY = reducedY [order(reducedY, decreasing=F)]
  x = 1:length(reducedY)
  
  plot(x, reducedY, pch=19, col="black")
  text(x, reducedY, names(reducedY))
  title(main = paste("Factor", theFactor, sep = " "))
  
  fileName = paste("Factor_", theFactor, ".csv", sep = "")
  write.csv(round(reducedY[order(reducedY, decreasing=T)], 3), fileName)
  classes[[theFactor]] = names(reducedY)
  
  print(paste("Factor ", theFactor, ": ", sep = ""))
  print(reducedY[order(reducedY, decreasing=T)])
  
}

# Clases que aparecen en los factores luego de cutoff:

classesPresent = unique(unlist(classes, use.names = FALSE))
classesPresent = classesPresent[order(classesPresent, decreasing=F)]

# Clases que quedan fuera de cualquier factor una vez que se ha
# aplicado este cutoff

classesNotPresent = setdiff (names(grades1), classesPresent)
classesNotPresent

# Para correr el experimento de LAK, las variables deben ser:
# grades = data1
# numbersOfFactos = 5
# cuttOff = 0.3

# Para correr el segundo análisis que envié, las variables deben ser:
# grades = data2
# numbersOfFactos = 6
# cuttOff = 0.3

# * El cutoff es un parámetro que uno elige.