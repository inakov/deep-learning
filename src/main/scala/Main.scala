import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics._
import datasets.housing.MNIST
import deep.learning.networks.Network

/**
 * Created by inakov on 19.07.15.
 */
object Main {

  def main(args: Array[String]) {

//    val m = DenseMatrix((0.54315592,  0.31597731), (0.68744078,  0.58687109))
//    val m2 = DenseMatrix((0.04315592,  0.31597731), (0.68744078,  0.58687109))
//    println(m. - m2)

    val mnistDataset = MNIST.loadTrainingDataSet
    val trainingData = mnistDataset._1

    val validationData = mnistDataset._2

    val net = new Network(List(784, 30, 10))
    net.SGD(trainingData, 30, 10, 0.5, testData = validationData)


  }

}
