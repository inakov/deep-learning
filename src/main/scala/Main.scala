import datasets.housing.MNIST

/**
 * Created by inakov on 19.07.15.
 */
object Main {

  def main(args: Array[String]) {
    val data = MNIST.read()

    MNIST.draw(data(180)._1, data(180)._2)
  }

}
