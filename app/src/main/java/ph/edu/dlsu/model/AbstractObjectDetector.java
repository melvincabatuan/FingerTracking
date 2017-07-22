package ph.edu.dlsu.model;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

/**
 * Created by cobalt on 6/21/16.
 */

public abstract class AbstractObjectDetector {

    abstract void setMinObjectSize(int size);
    abstract void start();
    abstract void detect(Mat imageGray, MatOfRect faces);
    abstract void stop();
    abstract void release();
}
