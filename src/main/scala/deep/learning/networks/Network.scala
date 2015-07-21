package deep.learning.networks

import breeze.linalg.{argmax, sum, DenseVector, DenseMatrix}
import breeze.numerics._
import breeze.stats.distributions.Uniform

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


/**
 * Created by inakov on 7/20/15.
 */
class Network(networkDefinition: List[Int]) {

  val numberOfLayers: Int = networkDefinition.length
  val definition: List[Int] = networkDefinition

  var biases: ArrayBuffer[DenseVector[Double]] = new ArrayBuffer[DenseVector[Double]]()
  var weights: ArrayBuffer[DenseMatrix[Double]] = new ArrayBuffer[DenseMatrix[Double]]()

  for(y <- definition.drop(1).take(numberOfLayers-1)) biases += DenseVector.rand[Double](y, rand = breeze.stats.distributions.Gaussian(0,1))

  /*
  * weights is initialized with uniformely sampled
  * from sqrt(-6./(inputsNum+outputsNum)) and sqrt(6./(inputsNum+outputsNum))
  * optimized for tanh activation function
  */
  for((inputsNum, outputsNum) <- definition.take(numberOfLayers-1).zip(definition.drop(1).take(numberOfLayers-1))){
    val low: Double = (-Math.sqrt(6.0 / (inputsNum + outputsNum))) * 4.0
    val high: Double = (Math.sqrt(6.0 / (inputsNum + outputsNum))) * 4.0
    val rng = Uniform(low, high)
    weights += DenseMatrix.rand[Double](inputsNum, outputsNum, rng)
  }

  /*
   * L1 norm ; one regularization option is to enforce L1 norm to
   * be small
   */
  def regularizationL1Norm(): Double = {
    weights.map(weight => sum(abs(weight))).sum
  }

  /*
  * square of L2 norm  one regularization option is to enforce
  * square of L2 norm to be small
  */
  def regularizationL2Norm(): Double = {
    weights.map(weight => sum(pow(weight, 2))).sum
  }

  def feedforward(a: DenseVector[Double]) = {
    var result = a

    for((b, w) <- biases.zip(weights))
      result = sigmoid((w.t * result) + b)
    result
  }

  def SGD(trainingData: Seq[(DenseVector[Double],DenseVector[Double])], epochs: Int, miniBatchSize: Int, eta: Double, lmbda: Double = 0.0, testData:Seq[(DenseVector[Double],DenseVector[Double])] = Nil): Unit ={
    val trainingDataLength: Int = trainingData.length
    for(j <- 0 until epochs){
      Random.shuffle(trainingData)
      val miniBatches: List[Seq[(DenseVector[Double],DenseVector[Double])]] =
        for(k <- (0 until trainingDataLength by miniBatchSize).toList) yield trainingData.slice(k, k + miniBatchSize)

      for(miniBatch <- miniBatches)
        updateMiniBatch(miniBatch, eta, lmbda, trainingDataLength)

      if (testData != Nil)
        println("Epoch " + j + ": " + evaluate(testData) + "/" + testData.length)
      else
        print("Epoch "+ j + " complete")
    }
  }

  def updateMiniBatch(miniBatch: Seq[(DenseVector[Double],DenseVector[Double])], eta: Double, lmbda: Double, n: Int): Unit = {
    var nabla_biases: ArrayBuffer[DenseVector[Double]] = biases.map(b => DenseVector.zeros[Double](b.length))
    var nabla_weights: ArrayBuffer[DenseMatrix[Double]] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    for((x, y) <- miniBatch){
      val backpropResult = backprop(x, y)
      val delta_nabla_b: ArrayBuffer[DenseVector[Double]] = backpropResult._1
      val delta_nabla_w: ArrayBuffer[DenseMatrix[Double]] = backpropResult._2

      nabla_biases = (nabla_biases, delta_nabla_b).zipped.map(_ + _)
      nabla_weights = (nabla_weights, delta_nabla_w).zipped.map(_ + _)
    }
    weights = (weights, nabla_weights).zipped.map((w, nw) => (1-eta*(lmbda/n))*w-(eta/miniBatch.length)*nw) //w-(eta/miniBatch.length)*nw
    biases = (biases, nabla_biases).zipped.map((b, nb) => b-(eta/miniBatch.length)*nb)
  }


  def backprop(x: DenseVector[Double], y: DenseVector[Double]):(ArrayBuffer[DenseVector[Double]], ArrayBuffer[DenseMatrix[Double]]) ={
    val nabla_b: ArrayBuffer[DenseVector[Double]] = biases.map(b => DenseVector.zeros[Double](b.length))
    val nabla_w: ArrayBuffer[DenseMatrix[Double]] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    var activation: DenseVector[Double] = x
    val activations: ArrayBuffer[DenseVector[Double]] = ArrayBuffer[DenseVector[Double]](x)
    var zs: ArrayBuffer[DenseVector[Double]] = ArrayBuffer[DenseVector[Double]]()

    //feedforward
    for((b, w) <- biases.zip(weights)){
      val z = (w.t * activation) + b
      zs += z
      activation = sigmoid(z)
      activations += activation
    }
    //backward pass
    var delta = costDerivative(activations.last, y) :* sigmoidPrime(zs.last)
    nabla_b(nabla_b.length-1) = delta
    nabla_w(nabla_w.length-1) = activations(activations.length-2) * delta.t
    for(l <- 2 until numberOfLayers){
      val z = zs(zs.length-l)
      val spv = sigmoidPrime(z);
      delta = (weights(weights.length-l+1) * delta) :* spv
      nabla_b(nabla_b.length-l) = delta
      nabla_w(nabla_w.length-l) = activations(activations.length-l-1) * delta.t
    }

    (nabla_b, nabla_w)
  }

  def evaluate(testData: Seq[(DenseVector[Double],DenseVector[Double])]) = {

    val testResult:Seq[(Int, Int)] = for((x,y) <- testData) yield (argmax(feedforward(x)), argmax(y))
    var result = 0;
    for((x,y) <- testResult; if y == x){
      result+=1;
    }
    result
  }

  def costDerivative(outputActivations: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = {
    outputActivations-y
  }

  def sigmoidPrime(z: DenseVector[Double]): DenseVector[Double]= {
    (sigmoid(z):*(1.0-sigmoid(z)))
  }

}
