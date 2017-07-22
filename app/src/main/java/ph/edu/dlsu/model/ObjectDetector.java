package ph.edu.dlsu.model;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import ph.edu.dlsu.mhealth.vision.DetectionBasedTracker;

import static android.content.ContentValues.TAG;

/**
 * Created by cobalt on 6/21/16.
 */

public final class ObjectDetector extends AbstractObjectDetector {

    private DetectionBasedTracker mNativeDetector;

    public ObjectDetector(File cascadeDir, InputStream is, String filename) {
        File mCascadeFile;
        try {

            mCascadeFile = new File(cascadeDir, filename);

            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }


    }

    @Override
    public void setMinObjectSize(int size) {
        mNativeDetector.setMinFaceSize(size);
    }

    @Override
    public void start() {
        mNativeDetector.start();
    }

    @Override
    public void detect(Mat imageGray, MatOfRect objects) {
        mNativeDetector.detect(imageGray, objects);
    }

    @Override
    public void stop() {
        mNativeDetector.stop();
    }

    @Override
    public void release() {
        mNativeDetector.release();
    }
}
