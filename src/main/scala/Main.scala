import breeze.linalg.{DenseVector, DenseMatrix, sum}
import breeze.numerics._
import datasets.housing.MNIST
import deep.learning.networks.Network

/**
 * Created by inakov on 19.07.15.
 */
object Main {

  def main(args: Array[String]) {
    val mnistDataset = MNIST.loadTrainingDataSet
    val trainingData = mnistDataset._1

    val validationData = mnistDataset._2

    val net = new Network(List(784, 30, 10))
    net.SGD(trainingData, 200, 30, 0.5, lmbda = 0.5, testData = validationData)

  }

}
