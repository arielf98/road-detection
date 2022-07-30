package org.tensorflow.lite.examples.detection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.hardware.Camera;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

public class ImageDetect extends AppCompatActivity {
    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] permissionstorage = {Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};
    private static final int PERMISSIONS_REQUEST = 1;
    private static final int PERMISSION_REQUEST_STORAGE = 112;
    private static final String TAG = "image";
    private static  final int CAMERA_REQUEST_CODE = 2;

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detect);

        detectButton = findViewById(R.id.detectButton);
        imageView = findViewById(R.id.imageViewDetect);
        textView = findViewById(R.id.classText);
        takePicture = findViewById(R.id.takePictureBtn);
         screenShoot = findViewById(R.id.saveButton);

        detectButton.setOnClickListener(v -> {
            if(this.imageView.getDrawable() != null){
                Handler handler = new Handler();
                new Thread(() -> {
                    final List<Classifier.Recognition> result = detector.recognizeImage(cropBitmap);
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            handleResult(cropBitmap, result, textView);
                        }
                    });
                }).start();
            } else {
                Toast.makeText(ImageDetect.this, "silahkan ambil gambar terlebih dahulu", Toast.LENGTH_SHORT).show();
            }
        });
        screenShoot.setOnClickListener(v -> {
            if(this.imageView.getDrawable() != null){
                takeScreenshot(getWindow().getDecorView().getRootView(), ImageDetect.this);
            } else {
                Toast.makeText(ImageDetect.this, "silahkan ambil gambar terlebih dahulu", Toast.LENGTH_SHORT).show();
            }
        });

        if(hasPermission()){
            takePicture.setOnClickListener(v -> {
                openCamera(ImageDetect.this);
            });
        } else {
            requestPermission();
        }
        requestStoragePermission(this);
        initBox();
    }

    private static final Logger LOGGER = new Logger();
    public static final int TF_OD_API_INPUT_SIZE = 416;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "yolov4-tiny-416.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;

    protected int previewWidth = 0;
    protected int previewHeight = 0;

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;

    private ImageView imageView;
    private Button detectButton, takePicture, screenShoot;
    private Classifier detector;
    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;
    private TextView textView;
    private File destination;

    private void initBox(){
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                sensorOrientation, MAINTAIN_ASPECT );

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(canvas -> tracker.draw(canvas));
        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);

        try {
            detector = YoloV4Classifier.create(
                    getAssets(),
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e){
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast = Toast.makeText(
                    getApplicationContext(),
                    "Classifier could not be initialized",
                    Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results, TextView textView){
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognition = new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results){
            final RectF location = result.getLocation();
            if(location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API){
                float percentage = (result.getConfidence()) * 100;
                String label = result.getTitle() + " " + String.format("%.0f%%", percentage);
                canvas.drawRect(location, paint);
                textView.setText(label);
//                cropToFrameTransform.mapRect(location);
//                result.setLocation(location);
//                mappedRecognition.add(result);
            }
        }

//        tracker.trackResults(mappedRecognition, new Random().nextInt());
//        trackingOverlay.postInvalidate();
        imageView.setImageBitmap(bitmap);

    }

    private void openCamera(Context context){
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        textView.setText("");
        this.imageView.setImageBitmap(null);
        this.imageView.setImageDrawable(null);
        startActivityForResult(camera, CAMERA_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode != RESULT_CANCELED){
            if(requestCode == CAMERA_REQUEST_CODE && resultCode == RESULT_OK && data != null){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                this.cropBitmap = Utils.processBitmap(image, TF_OD_API_INPUT_SIZE);
                this.imageView.setImageBitmap(cropBitmap);
            }
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                                ImageDetect.this,
                                "Camera permission is required",
                                Toast.LENGTH_LONG)
                        .show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
        }
    }

    private void requestStoragePermission(Activity activity){
        int permissions = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        // If storage permission is not given then request for External Storage Permission
        if (permissions != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions((Activity) activity, permissionstorage, PERMISSION_REQUEST_STORAGE);
    }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    protected void takeScreenshot(View view, final Context context) {
        Date date = new Date();
//        try {
            String dirpath;
            // Initialising the directory of storage
            dirpath= ImageDetect.this.getExternalFilesDir(Environment.DIRECTORY_PICTURES).toString();
            File file = new File(dirpath);
            if (!file.exists()) {
                boolean mkdir = file.mkdir();
            }
            // File name : keeping file name unique using data time.
            String path = dirpath + "/"+ date.getTime() + ".jpeg";
            view.setDrawingCacheEnabled(true);
            Bitmap bitmap = Bitmap.createBitmap(view.getDrawingCache());
            view.setDrawingCacheEnabled(false);
//            File imageurl = new File(path);
//            FileOutputStream outputStream = new FileOutputStream(imageurl);
//            bitmap.compress(Bitmap.CompressFormat.JPEG, 50, outputStream);
            MediaStore.Images.Media.insertImage(getContentResolver(), bitmap, path, path);
//            outputStream.flush();
//            outputStream.close();
            Log.d(TAG, "takeScreenshot Path: "+ path);
            Toast.makeText(ImageDetect.this,"image saved",Toast.LENGTH_LONG).show();
//            return imageurl;
//        } catch (FileNotFoundException io) {
//            io.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        return null;
    }
}