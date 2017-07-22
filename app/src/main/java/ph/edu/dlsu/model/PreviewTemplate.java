package ph.edu.dlsu.model;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

import ph.edu.dlsu.mhealth.android.Constants;

/**
 * Created by cobalt on 6/20/16.
 */

public final class PreviewTemplate {

    public static int NUMBER_OF_ZONES = 9;

    private int radiusGlobal;
    private int radiusNPzone;
    private Point globalCenter;
    private ArrayList<Point> zoneCenters;
    private ArrayList<Rect> zoneRoi;

    private Rect displayWindow;
    private int displayRadius;
    private Mat windowMask;

    private static int DELTA = 8;   // in-between zone delta
    private static final int globalThickness = 4;
    private static final int npThickness = 3;

    public void initCenters(int rows, int cols) {

        radiusGlobal = Math.min(rows, cols) / 2;
        radiusNPzone = radiusGlobal / 2;

        //globalCenter = new Point(cols / 2, rows / 2);
        globalCenter = new Point(radiusGlobal, rows / 2);

        // Initializing zone centers
        zoneCenters = new ArrayList<Point>(NUMBER_OF_ZONES);

        zoneCenters.add(0, new Point(globalCenter.x, globalCenter.y - radiusNPzone)); // 12 o'clock
        zoneCenters.add(1, new Point(globalCenter.x + radiusNPzone - DELTA, globalCenter.y - radiusNPzone + DELTA)); // 1-2 o'clock
        zoneCenters.add(2, new Point(globalCenter.x + radiusNPzone, globalCenter.y)); // 3 o'clock
        zoneCenters.add(3, new Point(globalCenter.x + radiusNPzone - DELTA, globalCenter.y + radiusNPzone - DELTA)); // 4-5 o'clock
        zoneCenters.add(4, new Point(globalCenter.x, globalCenter.y + radiusNPzone)); // 6 o'clock
        zoneCenters.add(5, new Point(globalCenter.x - radiusNPzone + DELTA, globalCenter.y + radiusNPzone - DELTA)); // 7-8 o'clock
        zoneCenters.add(6, new Point(globalCenter.x - radiusNPzone, globalCenter.y)); // 9 o'clock
        zoneCenters.add(7, new Point(globalCenter.x - radiusNPzone + DELTA, globalCenter.y - radiusNPzone + DELTA)); // 10-11 o'clock
        zoneCenters.add(8, globalCenter); // NP - central zone

        // initDisplayWindow
        int xTop = (int) zoneCenters.get(2).x + radiusNPzone + DELTA;
        int xBottom = cols - DELTA;
        displayRadius = (xBottom - xTop) / 2;
        displayWindow = new Rect(new Point(xTop, globalCenter.y - displayRadius), new Point(xBottom, globalCenter.y + displayRadius));
    }

    public void initMask(){
        windowMask = Mat.zeros(displayWindow.height, displayWindow.width, CvType.CV_8UC1);
        Imgproc.circle(windowMask, new Point(windowMask.cols()/2, windowMask.rows()/2), displayRadius, Constants.WHITE, -1); // negative thickness means filled circle
    }

    public Mat getWindowMask(){
        return windowMask;
    }

    public void initRois(int height) {
        // Initializing zone rois
        zoneRoi = new ArrayList<Rect>(NUMBER_OF_ZONES);
        for (int i = 0; i < NUMBER_OF_ZONES; i++) {
            int xTemp = Math.max(0, (int) zoneCenters.get(i).x - radiusNPzone);
            int yTemp = Math.max(0, (int) zoneCenters.get(i).y - radiusNPzone);
            int heightTemp = Math.min(2 * radiusNPzone, height - yTemp);
            zoneRoi.add(i, new Rect(xTemp, yTemp, 2 * radiusNPzone, heightTemp));
        }
    }


    public void displayTemplate(Mat src) {

        /// Display target circle with center point and outline
        Imgproc.circle(src, globalCenter, radiusNPzone, Constants.GREEN, npThickness/3, 8, 0);
        Imgproc.circle(src, globalCenter, radiusGlobal - DELTA, Constants.GREEN, globalThickness/3, 4, 0);
        /// 4- and 8-connected lines; 0 for the shift

        /// Draw the lines
        Imgproc.line(src, new Point(globalCenter.x - radiusGlobal, globalCenter.y), new Point(globalCenter.x - radiusNPzone, globalCenter.y), Constants.GREEN, npThickness);
        Imgproc.line(src, new Point(globalCenter.x, globalCenter.y - radiusGlobal), new Point(globalCenter.x, globalCenter.y - radiusNPzone), Constants.GREEN, npThickness);
        Imgproc.line(src, new Point(globalCenter.x + radiusGlobal, globalCenter.y), new Point(globalCenter.x + radiusNPzone, globalCenter.y), Constants.GREEN, npThickness);
        Imgproc.line(src, new Point(globalCenter.x, globalCenter.y + radiusGlobal), new Point(globalCenter.x, globalCenter.y + radiusNPzone), Constants.GREEN, npThickness);
    }

    public void displayAllZone(Mat src) {
        for (int i = 0; i < NUMBER_OF_ZONES; i++) {
            Imgproc.circle(src, zoneCenters.get(i), radiusNPzone, Constants.GREEN, npThickness, 8, 0);
        }
    }

    public void displayZone(Mat src, int zoneIndex) {
        if (zoneIndex >= 0 && zoneIndex < NUMBER_OF_ZONES)
            Imgproc.circle(src, zoneCenters.get(zoneIndex), radiusNPzone, Constants.WHITE, npThickness, 8, 0);
    }

    public Point getZoneCenter(int zoneIndex) {
        return zoneCenters.get(zoneIndex);
    }

    public Point getGlobalCenter() {
        return globalCenter;
    }

    public Rect getRoi(int zoneIndex) {
        return zoneRoi.get(zoneIndex);
    }

    public Rect getDisplayWindow() {
        return displayWindow;
    }

    public int getRadiusNPzone() {
        return radiusNPzone;
    }

    public int getRadiusGlobal() {
        return radiusGlobal;
    }
}