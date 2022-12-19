# Databricks notebook source
import os

bank = spark.read.format('csv').option('header','true').option('inferSchema','true').load(f"file:{os.getcwd()}/bank.csv")
bank.display()

# COMMAND ----------

bank.count()

# COMMAND ----------

bank.groupBy('deposit').count().display()

# COMMAND ----------

bank.groupBy('job').count().display()

# COMMAND ----------

display(bank.select('marital','balance'))

# COMMAND ----------

bank.dtypes

# COMMAND ----------

numeric_features = [t[0] for t in bank.dtypes if t[1] == 'int']
display(bank.select(numeric_features).describe())

# COMMAND ----------

stages = []
categoricalColumns = ['job','marital','education','default','housing','loan','contact','poutcome']

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder

for categoricalCol in categoricalColumns:

    indexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')

    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()],outputCols=[categoricalCol + 'classVec'])

    stages += [indexer,  encoder]

# COMMAND ----------

label_stringIndexer = StringIndexer(inputCol = 'deposit', outputCol='label')

stages += [label_stringIndexer]

# COMMAND ----------

numericCols = ['age','balance','duration','campaign','pdays','previous']

assemblerInputs = [c + 'classVec' for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols = assemblerInputs, outputCol='originalFeatures')

stages += [assembler]

# COMMAND ----------

from pyspark.ml.feature import UnivariateFeatureSelector

selector = UnivariateFeatureSelector(selectionMode='numTopFeatures', featuresCol='originalFeatures',outputCol='features',labelCol='label')
selector.setFeatureType('continuous').setLabelType('categorical').setSelectionThreshold(20)

# COMMAND ----------

stages+= [selector]

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(bank)

# COMMAND ----------

bank_transformed = pipelineModel.transform(bank)
bank_transformed.select('features','label').display()

# COMMAND ----------

stages

# COMMAND ----------

bank_train, bank_test = bank_transformed.randomSplit([0.7,0.3], seed = 2018)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# COMMAND ----------

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth =3)
dtModel = dt.fit(bank_train)

# COMMAND ----------

dtModel.numNodes, dtModel.depth

# COMMAND ----------

display(dtModel)

# COMMAND ----------

dtPreds = dtModel.transform(bank_test)

dtPreds.select('age','job','rawPrediction','prediction','probability','label').display()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

dtEval = BinaryClassificationEvaluator()

# COMMAND ----------

dtEval.evaluate(dtPreds)

# COMMAND ----------

# hyperparameter tuning with cross validation

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder().addGrid(dt.maxDepth,[1,3,6,10]).addGrid(dt.maxBins,[20,40,60,80]).build())

# COMMAND ----------

cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=dtEval, numFolds=5)

# COMMAND ----------

cvModel = cv.fit(bank_train) 

# COMMAND ----------

cvModel.bestModel.numNodes, cvModel.bestModel.depth

# COMMAND ----------

cvPreds = cvModel.transform(bank_test)

cvPreds.select('label','prediction').display()

# COMMAND ----------

dtEval.evaluate(cvPreds)

# COMMAND ----------


