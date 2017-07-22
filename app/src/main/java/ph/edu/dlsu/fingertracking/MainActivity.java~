package ph.edu.dlsu.fingertracking;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageButton;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.atomic.AtomicReference;

import butterknife.Bind;
import butterknife.ButterKnife;
import ph.edu.dlsu.mhealth.android.Constants;
import ph.edu.dlsu.mhealth.vision.CMT;
import ph.edu.dlsu.mhealth.vision.Camshift;
import ph.edu.dlsu.mhealth.vision.CustomTracker;
import ph.edu.dlsu.mhealth.vision.KCF;
import ph.edu.dlsu.model.ObjectDetector;
import ph.edu.dlsu.model.PreviewTemplate;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "tag.FpTracking";

    public static final String PHOTO_FILE_EXTENSION = ".png";
    public static final String PHOTO_MIME_TYPE = "image/png";
    public static final String EXTRA_PHOTO_URI =
            "ph.edu.dlsu.fingertracking.MainActivity.extra.PHOTO_URI";
    public static final String EXTRA_PHOTO_DATA_PATH =
            "ph.edu.dlsu.fingertracking.MainActivity.extra.PHOTO_DATA_PATH";

    public static final int VIEW_MODE_RGB = 0;
    public static final int VIEW_MODE_KCF = 1;
    public static final int VIEW_MODE_MIL = 2;
    public static final int VIEW_MODE_TLD = 3;
    public static final int VIEW_MODE_BOOST = 4;
    public static final int VIEW_MODE_MEDIAN = 5;
    public static final int VIEW_MODE_CAMSHIFT = 6;
    public static final int VIEW_MODE_COLOR = 7;
    public static final int VIEW_MODE_CMT = 8;
    public static int viewMode = VIEW_MODE_CMT;    // default

    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediate;
    private Mat mMask;

    private final float RELATIVE_OBJECT_SIZE = 0.2f;
    private int mAbsoluteObjectSize = 0;

    @Bind(R.id.fpt_activity_surface_view)
    CameraBridgeViewBase mOpenCvCameraView;

    @Bind(R.id.btn_next_fpt_activity)
    ImageButton nextBtn;

    private ObjectDetector fingerpadDetector;

    private PreviewTemplate previewTemplate;
    private int zoneCounter;
    private int currentZone;
    private Rect zoomWindow;


    // Tracking
    private CMT cmt;
    private KCF kcf;
    private CustomTracker customTracker;
    private Camshift camshift;
    private boolean isInitialized;


    // ROI Selection
    SurfaceHolder _holder;
    private int _canvasImgYOffset;
    private int _canvasImgXOffset;
    private Rect _trackedBox = null;


    // Tracking accuracy testing
    private boolean isFreezeCamera;
    private int frameCounter;

    // Take picture view
    private Bitmap photo;


    static {

        // Load libopencv_java3.so located at jniLibs directory
        System.loadLibrary("opencv_java3");

        // Load native library after(!) OpenCV initialization
        System.loadLibrary("mhealth_vision");
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);

        initCascadeClassifier();

        nextBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                zoneCounter++;
                currentZone = zoneCounter % PreviewTemplate.NUMBER_OF_ZONES;
                ph.edu.dlsu.mhealth.android.Utils.showToast(MainActivity.this, "palpate zone " + (currentZone), R.mipmap.ic_launcher);
            }
        });

        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableFpsMeter();
        _holder = mOpenCvCameraView.getHolder();

        final AtomicReference<Point> trackedBox1stCorner = new AtomicReference<>();

        final Paint rectPaint = new Paint();
        rectPaint.setColor(Color.rgb(0, 255, 0));
        rectPaint.setStrokeWidth(5);
        rectPaint.setStyle(Paint.Style.STROKE);


        mOpenCvCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                // re-init
                final Point corner = new Point(
                        event.getX() - _canvasImgXOffset, event.getY()
                        - _canvasImgYOffset);
                switch (event.getAction()) {

                    case MotionEvent.ACTION_DOWN:
                        trackedBox1stCorner.set(corner);
                        Log.i(TAG, "1st corner: " + corner);
                        break;

                    case MotionEvent.ACTION_UP:
                        _trackedBox = new Rect(trackedBox1stCorner.get(), corner);
                        if (_trackedBox.area() > 100) {
                            Log.i(TAG, "Tracked box DEFINED: " + _trackedBox);
                            isInitialized = false;
                        } else
                            _trackedBox = null;
                        break;

                    case MotionEvent.ACTION_MOVE:
                        final android.graphics.Rect rect = new android.graphics.Rect(
                                (int) trackedBox1stCorner.get().x
                                        + _canvasImgXOffset,
                                (int) trackedBox1stCorner.get().y
                                        + _canvasImgYOffset, (int) corner.x
                                + _canvasImgXOffset, (int) corner.y
                                + _canvasImgYOffset);
                        final Canvas canvas = _holder.lockCanvas(rect);
                        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

                        canvas.drawRect(rect, rectPaint);
                        _holder.unlockCanvasAndPost(canvas);

                        break;
                }

                return true;
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        isInitialized = false;
        _trackedBox = null;
    }

    @Override
    public void onResume() {
        super.onResume();
        mOpenCvCameraView.enableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        fingerpadDetector.release();

        releaseActiveTrackers();
    }

    private void releaseActiveTrackers() {

        if (cmt != null) {
            cmt.release();
        } else if (kcf != null) {
            kcf.release();
        } else if (customTracker != null) {
            customTracker.release();
        } else if (camshift != null) {
            camshift.release();
        }

        // Release roi box
        if (_trackedBox != null) {
            _trackedBox = null;
        }

        // Reset
        isInitialized = false;
    }

    public void onCameraViewStarted(int width, int height) {
        // Initialize Matrices for Images
        mGray = new Mat();
        mRgba = new Mat();
        mIntermediate = new Mat();

        // Initialize the Object Template
        previewTemplate = new PreviewTemplate();
        previewTemplate.initCenters(height, width);
        previewTemplate.initRois(height);
        previewTemplate.initMask();

        mMask = previewTemplate.getWindowMask();  // circular mask

        //currentZone = Constants.TWELVEoCLOCK_ZONE;
        currentZone = Constants.NP_CENTRAL_ZONE;

        // Initialize zoom window
        zoomWindow = previewTemplate.getDisplayWindow();

        // Begin detection and set minimum detectable object
        fingerpadDetector.start();
        if (Math.round(height * RELATIVE_OBJECT_SIZE) > 0) {
            mAbsoluteObjectSize = Math.round(height * RELATIVE_OBJECT_SIZE);
        }
        fingerpadDetector.setMinObjectSize(mAbsoluteObjectSize);

        // Initialize accuracy computation parameter
        frameCounter = 0;
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mMask.release();
        mIntermediate.release();
        fingerpadDetector.stop(); // To be released on Activity Destroy
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // Show template for initial zone
        previewTemplate.displayZone(mRgba, currentZone);

        // Acquire the region of interest
        Rect roi = previewTemplate.getRoi(currentZone);

        MatOfRect objects = new MatOfRect();

        // Detect the finger pads in the zoomed image
        if (fingerpadDetector != null)
            fingerpadDetector.detect(mGray.submat(roi), objects);

        Rect[] objectsArray = objects.toArray();
        for (Rect aObjectsArray : objectsArray) {
            // Correction of rect location by offset
            Point topLeft = new Point(roi.x + aObjectsArray.tl().x, roi.y + aObjectsArray.tl().y);
            Point bottomRight = new Point(roi.x + aObjectsArray.br().x, roi.y + aObjectsArray.br().y);
            Imgproc.rectangle(mRgba, topLeft, bottomRight, Constants.GREEN, 3);
        }

        // Only apply tracking if roi box has been chosen
        if (_trackedBox != null) {

            // xBox and yBox wrt roi
            long xBox = Math.max(0, _trackedBox.x - roi.x);
            long yBox = Math.max(0, _trackedBox.y - roi.y);
            long wBox = Math.min(_trackedBox.width, (int) roi.br().x - _trackedBox.x);
            long hBox = Math.min(_trackedBox.height, (int) roi.br().y - _trackedBox.y);

            // Note _trackedBox is with respect to global coordinates
            Mat tempGray = mGray.submat(roi);

            // Increase contrast
            Imgproc.equalizeHist(tempGray, tempGray);

            // Initialize and apply various trackers
            switch (MainActivity.viewMode) {

                case MainActivity.VIEW_MODE_CMT:
                    if (!isInitialized) {

                        // Attempt to initialize with the chosen ROI, but return false
                        // when there aren't enough keypoints
                        isInitialized = cmt.initialize(tempGray, xBox, yBox, wBox, hBox);
                        if (isInitialized) {
                            Log.d(TAG, "Successfully initialized!");
                        } else {
                            Log.d(TAG, "Not enough keypoints! Try again!");
                        }


                    } else { // If already initialized, then simply apply continuously
                        cmt.apply(tempGray, mRgba.submat(roi));
                    }
                    break;


                case MainActivity.VIEW_MODE_KCF:
                    if (!isInitialized) {

                        // Attempt to initialize with the chosen ROI, but return false
                        // when there aren't enough keypoints
                        isInitialized = kcf.initialize(mRgba.submat(roi), xBox, yBox, wBox, hBox);
                        if (isInitialized) {
                            Log.d(TAG, "Successfully initialized!");
                        } else {
                            Log.d(TAG, "Not enough keypoints! Try again!");
                        }


                    } else { // If already initialized, then simply apply continuously
                        kcf.apply(mRgba.submat(roi), mRgba.submat(roi));
                    }
                    break;

                case MainActivity.VIEW_MODE_MIL:
                case MainActivity.VIEW_MODE_BOOST:
                case MainActivity.VIEW_MODE_MEDIAN:
                case MainActivity.VIEW_MODE_TLD:
                case MainActivity.VIEW_MODE_COLOR:

                    if (!isInitialized) {

                        isInitialized = customTracker.initialize(mRgba.submat(roi), xBox, yBox, wBox, hBox);
                        if (isInitialized) {
                            Log.d(TAG, "Successfully initialized!");
                        } else {
                            Log.d(TAG, "Not enough keypoints! Try again!");
                        }


                    } else { // If already initialized, then simply apply continuously
                        customTracker.apply(mRgba.submat(roi), mRgba.submat(roi));
                    }
                    break;

                case MainActivity.VIEW_MODE_CAMSHIFT:

                    if (!isInitialized) {

                        isInitialized = camshift.initialize(mRgba.submat(roi), xBox, yBox, wBox, hBox);
                        if (isInitialized) {
                            Log.d(TAG, "Successfully initialized!");
                        } else {
                            Log.d(TAG, "Not initialzed! Try again!");
                        }


                    } else { // If already initialized, then simply apply continuously
                        camshift.apply(mRgba.submat(roi), mRgba.submat(roi));
                    }
                    break;

                case MainActivity.VIEW_MODE_RGB:
                default:
                    releaseActiveTrackers();
            }


        } // end tracked box

        // Resize roi and copy back to mRgba display circular window
        Mat zoomImageRgba = mRgba.submat(zoomWindow);
        Imgproc.resize(mRgba.submat(roi), mIntermediate, zoomImageRgba.size());
        mIntermediate.copyTo(zoomImageRgba, mMask);

        // Display object target
        previewTemplate.displayTemplate(mRgba);

        zoomImageRgba.release();
        mIntermediate.release();

        return mRgba;
    }

    // Initiates the Object detector
    private void initCascadeClassifier() {
        // load cascade file from application resources raw directory
        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalnp);
        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
        fingerpadDetector = new ObjectDetector(cascadeDir, is, "lbpcascade_frontalnp.xml");
        cascadeDir.delete();
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {

        MenuInflater inflater = getMenuInflater();

        /*********  Update this layout if you add more filters  **********/
        inflater.inflate(R.menu.activity_main_menu, menu);

        return super.onCreateOptionsMenu(menu);
    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        releaseActiveTrackers();

        long action = item.getItemId();

        if (action == R.id.action_rgb) {
            viewMode = VIEW_MODE_RGB;
        }else if (action == R.id.action_cmt) {
            cmt = new CMT();
            viewMode = VIEW_MODE_CMT;
        } else if (action == R.id.action_kcf) {
            kcf = new KCF();
            viewMode = VIEW_MODE_KCF;
        } else if (action == R.id.action_mil) {
            customTracker = new CustomTracker("MIL");
            viewMode = VIEW_MODE_MIL;
        } else if (action == R.id.action_tld) {
            customTracker = new CustomTracker("TLD");
            viewMode = VIEW_MODE_TLD;
        } else if (action == R.id.action_boost) {
            customTracker = new CustomTracker("BOOSTING");
            viewMode = VIEW_MODE_BOOST;
        } else if (action == R.id.action_median) {
            customTracker = new CustomTracker("MEDIANFLOW");
            viewMode = VIEW_MODE_MEDIAN;
        } else if (action == R.id.action_camshift) {
            camshift = new Camshift();
            viewMode = VIEW_MODE_CAMSHIFT;
        }

        return super.onOptionsItemSelected(item);
    }


    private void takePhoto(Mat rgba) {

        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String fileName = currentDateandTime + PHOTO_FILE_EXTENSION;

        photo = null;

        try {
            photo = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgba, photo);
        } catch (CvException e) {
            //Log.d(TAG, e.getMessage());
        }


        FileOutputStream out = null;
        final String appName = getString(R.string.app_name);
        final String albumPath = Environment.getExternalStorageDirectory() + File.separator + appName;
        final String photoPath = albumPath + File.separator + fileName;


        File sd = new File(albumPath);
        boolean success = true;
        if (!sd.exists()) {
            success = sd.mkdir();
        }

        if (success) {
            File dest = new File(sd, fileName);

            try {
                out = new FileOutputStream(dest);
                photo.compress(Bitmap.CompressFormat.PNG, 1, out); // bmp is your Bitmap instance
                // PNG is a lossless format, the compression factor (100) is ignored (changed to 0)

            } catch (Exception e) {
                // e.printStackTrace();
                //Log.d(TAG, e.getMessage());
            } finally {
                try {
                    if (out != null) {
                        out.close();
                        //Log.d(TAG, "OK!!");
                    }
                } catch (IOException e) {
                    //Log.d(TAG, e.getMessage() + "Error");
                    // e.printStackTrace();
                }
            }

            // Recycle bitmap
            if (photo != null)
                photo.recycle();

            photo = null;


            // Save photo information
            final ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DATA, photoPath);
            values.put(MediaStore.Images.Media.MIME_TYPE, PHOTO_MIME_TYPE);
            values.put(MediaStore.Images.Media.TITLE, appName);
            values.put(MediaStore.Images.Media.DESCRIPTION, appName);
            values.put(MediaStore.Images.Media.DATE_TAKEN, currentDateandTime);


            // Try to insert the photo into the MediaStore.
            Uri uri = null;

            try {
                uri = getContentResolver().insert(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            } catch (final Exception e) {
                //Log.e(TAG, "Failed to insert photo into MediaStore");
                e.printStackTrace();
                // Since the insertion failed, delete the photo.
                File fphoto = new File(photoPath);
                if (!fphoto.delete()) {
                    //Log.e(TAG, "Failed to delete non-inserted photo");
                }
            }
        }
    }
}
