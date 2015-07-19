package datasets.housing

import java.io.{FileInputStream, DataInputStream}

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.plot._

import scala.collection.mutable.ArrayBuffer

/**
 * Created by inakov on 19.07.15.
 */
object MNIST {

  val trainimagefile = getClass.getResource("/train-images-idx3-ubyte").getPath
  val trainlabelfile = getClass.getResource("/train-labels-idx1-ubyte").getPath

  def read(labelfile: String = trainlabelfile, imagefile : String = trainimagefile):Seq[(DenseVector[Double], DenseVector[Double])] ={
    val imagesArray = ArrayBuffer[DenseVector[Double]]()
    val labelsArray = ArrayBuffer[DenseVector[Double]]()
    val labels = new DataInputStream(new FileInputStream(labelfile))
    val images = new DataInputStream(new FileInputStream(imagefile))
    var magicNumber = labels.readInt();
    if (magicNumber != 2049) {
      System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)")
      System.exit(0)
    }
    magicNumber = images.readInt()
    if (magicNumber != 2051) {
      System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)")
      System.exit(0)
    }
    val numLabels = labels.readInt()
    val numImages = images.readInt()
    val numRows = images.readInt()
    val numCols = images.readInt()
    if (numLabels != numImages) {
      System.err.println("Image file and label file do not contain the same number of entries.")
      System.err.println("  Label file contains: " + numLabels)
      System.err.println("  Image file contains: " + numImages)
      System.exit(0);
    }

    val start = System.currentTimeMillis()
    var numLabelsRead = 0;
    var numImagesRead = 0;

    while (labels.available() > 0 && numLabelsRead < numLabels) {
      labelsArray += vectorizedLabel(labels.readUnsignedByte())
      numLabelsRead += 1
      val imageTmp: DenseMatrix[Double] = DenseMatrix.zeros[Double](numCols, numRows)
      for (colIdx <- 0 until numCols) {
        for (rowIdx <- 0 until numRows) {
          imageTmp(colIdx, rowIdx) = images.readUnsignedByte();
        }
      }
      imagesArray += imageTmp.toDenseVector
      numImagesRead += 1;

      if (numLabelsRead % 10 == 0) {
        System.out.print(".");
      }
      if ((numLabelsRead % 800) == 0) {
        System.out.print(" " + numLabelsRead + " / " + numLabels);
        val end = System.currentTimeMillis();
        val elapsed = end - start;
        val minutes = elapsed / (1000 * 60);
        val seconds = (elapsed / 1000) - (minutes * 60);
        System.out.println("  " + minutes + " m " + seconds + " s ");
      }

    }
    imagesArray.zip(labelsArray)
  }

  def vectorizedLabel(label: Int): DenseVector[Double] = {
    val result = DenseVector.zeros[Double](10)
    result(label) = 1.0

    result
  }

  def draw(imageVector: DenseVector[Double], label: DenseVector[Double], save: Boolean = true) = {
    println("Label: " + label)
    val f = Figure(name = "Label " + label.toString())
    f.subplot(0) += image(imageVector.toDenseMatrix.reshape(28, 28))

    if(save) f.saveas("image.png")
  }

}
