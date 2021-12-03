// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.agriculture.vision.mscgnet.activities.image.segmentation;

import android.Manifest;
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
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TableLayout;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.agriculture.vision.mscgnet.R;
import org.agriculture.vision.mscgnet.activities.object.detection.processing.ResultView;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class ImageSegmentationActivity extends AppCompatActivity implements Runnable,
//        View.OnClickListener,
        AdapterView.OnItemSelectedListener {
    private static final String TAG = "MainActivity";
    private int mImageIndex = 0;
    private Spinner modeSpinner;
    private ImageView mImageView;
    private ResultView mResultView;
    private TableLayout labelsToColorLegend;
    private int overlayMode = 0;
    private Button mButtonDetect, mButtonSegment;
    private ProgressBar mProgressBar;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    public static final String inputBasename = "input.jpg";
    public static final String lutRGBBasename = "lut_rgb.png";
    public static final String lutBasename = "lut.png";

    private Bitmap mBitmap = null;
    private Bitmap lutRGB = null;
    private Bitmap lutRGBOverlayedImage = null;
    private Module mModule = null;
    private static final int CLASSNUM = 10;

    // Label encoding
    public static final int BACKGROUND = 0;
    public static final int WATER = 1;
    public static final int DOUBLE_PLANT = 2;
    public static final int PLANTER_SKIP = 3;
    public static final int DRYDOWN = 4;
    public static final int WATERWAY = 5;
    public static final int WEED_CLUSTER = 6;
    public static final int ENDROW = 7;
    public static final int NUTRIENT_DEFICIENT = 8;
    public static final int STORM_DAMAGE = 9;

    // colors in HEX to be deferenced by the associated encoded label
    public static final int[] COLOR_MAP = {
            0xFFFFFF00, // water
            0xFFFF00FF, // double plant
            0xFF00FF00, // planter skip
            0xFF0000FF, // drydown
            0xFFFFFFFF, // waterway
            0xFF00FFFF, // weed cluster
            0xFF0080FF, // endrow
            0xFF800080, // nutrient defiency
            0xFFFF0000 // storm damage
    };


    // need color code of labels
    private final String[] lutRGBImages = {
            "mscg-samples/1E3FJWUF1_3911-3514-4423-4026" + "/" + lutRGBBasename,
            "mscg-samples/1FY8MBG8K_3050-1017-3562-1529" + "/" + lutRGBBasename,
            "mscg-samples/1FY8MBG8K_3645-665-4157-1177" + "/" + lutRGBBasename,
            "mscg-samples/1FY8MBG8K_10140-9620-10652-10132" + "/" + lutRGBBasename,
            "mscg-samples/1KK6Y8UVT_10337-3853-10849-4365" + "/" + lutRGBBasename,
            "mscg-samples/1KK6Y8UVT_11917-3225-12429-3737" + "/" + lutRGBBasename,
            "mscg-samples/1KK6Y8UVT_12429-3225-12941-3737" + "/" + lutRGBBasename,
            "mscg-samples/1NEDW647M_1011-7250-1523-7762" + "/" + lutRGBBasename,
            "mscg-samples/1NEDW647M_1020-9073-1532-9585" + "/" + lutRGBBasename,
    };

    private final String[] mLUTImages = {
            "mscg-samples/1E3FJWUF1_3911-3514-4423-4026" + "/" + lutBasename,
            "mscg-samples/1FY8MBG8K_3050-1017-3562-1529" + "/" + lutBasename,
            "mscg-samples/1FY8MBG8K_3645-665-4157-1177" + "/" + lutBasename,
            "mscg-samples/1FY8MBG8K_10140-9620-10652-10132" + "/" + lutBasename,
            "mscg-samples/1KK6Y8UVT_10337-3853-10849-4365" + "/" + lutBasename,
            "mscg-samples/1KK6Y8UVT_11917-3225-12429-3737" + "/" + lutBasename,
            "mscg-samples/1KK6Y8UVT_12429-3225-12941-3737" + "/" + lutBasename,
            "mscg-samples/1NEDW647M_1011-7250-1523-7762" + "/" + lutBasename,
            "mscg-samples/1NEDW647M_1020-9073-1532-9585" + "/" + lutBasename,
    };

    private final String[] mTestInputImages = {
            "mscg-samples/1E3FJWUF1_3911-3514-4423-4026" + "/" + inputBasename,
            "mscg-samples/1FY8MBG8K_3050-1017-3562-1529" + "/" + inputBasename,
            "mscg-samples/1FY8MBG8K_3645-665-4157-1177" + "/" + inputBasename,
            "mscg-samples/1FY8MBG8K_10140-9620-10652-10132" + "/" + inputBasename,
            "mscg-samples/1KK6Y8UVT_10337-3853-10849-4365" + "/" + inputBasename,
            "mscg-samples/1KK6Y8UVT_11917-3225-12429-3737" + "/" + inputBasename,
            "mscg-samples/1KK6Y8UVT_12429-3225-12941-3737" + "/" + inputBasename,
            "mscg-samples/1NEDW647M_1011-7250-1523-7762" + "/" + inputBasename,
            "mscg-samples/1NEDW647M_1020-9073-1532-9585" + "/" + inputBasename,
    };
//    private String[] sampleDirectories = {
//            "1E3FJWUF1_3911-3514-4423-4026",
//            "1FY8MBG8K_3050-1017-3562-1529",
//            "1FY8MBG8K_3645-665-4157-1177",
//            "1FY8MBG8K_10140-9620-10652-10132",
//            "1KK6Y8UVT_10337-3853-10849-4365",
//            "1KK6Y8UVT_11917-3225-12429-3737",
//            "1KK6Y8UVT_12429-3225-12941-3737",
//            "1NEDW647M_1011-7250-1523-7762",
//            "1NEDW647M_1020-9073-1532-9585"
//    };

//    private String[] mTestImages = {
//            "test1.png", "test2.jpg", "test3.png"
//    }


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, mTestInputImages.toString());

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);
        labelsToColorLegend = findViewById(R.id.labelsColorTable);
        modeSpinner = findViewById(R.id.displayModeSpinner);
        modeSpinner.setOnItemSelectedListener(this);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestInputImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        // place for rendering images from file
        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        // setup buttons
        final Button buttonNextTestImage = findViewById(R.id.testButton);
        buttonNextTestImage.setText((
                String.format("Test Image 1/%d", mTestInputImages.length)
        ));
        buttonNextTestImage.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestInputImages.length;
                buttonNextTestImage.setText(
                        String.format("Test Image %d/%d", mImageIndex + 1, mTestInputImages.length)
                );

                try {
                    mBitmap = BitmapFactory.decodeStream(
                            getAssets().open(mTestInputImages[mImageIndex])
                    );
                    // display input image
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });

        final Button buttonSelectFromDevice = findViewById(R.id.selectButton);
        buttonSelectFromDevice.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = {"Choose from Photos", "Take Picture", "Cancel"};
                AlertDialog.Builder builder = new AlertDialog.Builder(ImageSegmentationActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        } else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto, 1);
                        } else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button reloadTestImage = findViewById(R.id.reloadButton);
        reloadTestImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
//                mImageIndex = (mImageIndex + 1) % mTestInputImages.length;
//                buttonNextTestImage.setText(
//                        String.format("Test Image %d/%d", mImageIndex + 1, mTestInputImages.length)
//                );

                try {
                    mBitmap = BitmapFactory.decodeStream(
                            getAssets().open(mTestInputImages[mImageIndex])
                    );
                    // display input image
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonColorSegments = findViewById(R.id.segmentButton);
        buttonColorSegments.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mResultView.setVisibility(View.INVISIBLE);
//                mProgressBar.setVisibility(ProgressBar.VISIBLE);
//                mButtonSegment.setText(getString(R.string.run_model));

                // TODO: refactor to be handled by ImageSegmentationVisualization class implementing Runnable


                // apply post processing here
                try {
                    lutRGB = BitmapFactory.decodeStream(
//                            getAssets().open(mLUTImages[mImageIndex])
                            // access cached rendering of RGB rendering of model's LUT output
                            getAssets().open(lutRGBImages[mImageIndex])
                    );
                    // LUT only
                    if (overlayMode == 1) {
                        mImageView.setImageBitmap(lutRGB);
                    }
                    // LUT to black-rm LUT + input image
                    else if (overlayMode == 0) {
                        // remove black
                        Bitmap litRGBTransparent = null;
                        litRGBTransparent = createTransparentBitmapFromBitmap(lutRGB, 0xFF000000);
                        lutRGBOverlayedImage = overlay(mBitmap, litRGBTransparent);
                        // display cached RGB rendered segmentation
                        mImageView.setImageBitmap(lutRGBOverlayedImage);
                    }

                    // TODO: breaking rendering of segmentation
//                    Thread thread = new Thread(ImageSegmentationActivity.this);
//                    thread.start();


                    // show only lutRGB
//                    mImageView.setImageBitmap(litRGBTransparent);


                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


//        final Button buttonLive = findViewById(R.id.liveButton);
//        buttonLive.setOnClickListener(new View.OnClickListener() {
//            public void onClick(View v) {
//                final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
//                startActivity(intent);
//            }
//        });

//        mButtonDetect = findViewById(R.id.detectButton);
//        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
//        mButtonDetect.setOnClickListener(new View.OnClickListener() {
//            public void onClick(View v) {
//                mButtonDetect.setEnabled(false);
//                mProgressBar.setVisibility(ProgressBar.VISIBLE);
//                mButtonDetect.setText(getString(R.string.run_model));
//
//                mImgScaleX = (float) mBitmap.getWidth() / PrePostProcessor.mInputWidth;
//                mImgScaleY = (float) mBitmap.getHeight() / PrePostProcessor.mInputHeight;
//
//                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float) mImageView.getWidth() / mBitmap.getWidth() : (float) mImageView.getHeight() / mBitmap.getHeight());
//                mIvScaleY = (mBitmap.getHeight() > mBitmap.getWidth() ? (float) mImageView.getHeight() / mBitmap.getHeight() : (float) mImageView.getWidth() / mBitmap.getWidth());
//
//                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth()) / 2;
//                mStartY = (mImageView.getHeight() - mIvScaleY * mBitmap.getHeight()) / 2;
//
//                Thread thread = new Thread(MainActivity.this);
//                thread.start();
//            }
//        });
//        buttonColorSegments = findViewById(R.id.segment);


// Load model from file
//        try {
//            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "yolov5s.torchscript.ptl"));
//            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("aicook.txt")));
//            String line;
//            List<String> classes = new ArrayList<>();
//            while ((line = br.readLine()) != null) {
//                classes.add(line);
//            }
//            PrePostProcessor.mClasses = new String[classes.size()];
//            classes.toArray(PrePostProcessor.mClasses);
//        } catch (IOException e) {
//            Log.e("Object Detection", "Error reading assets", e);
//            finish();
//        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }

//    @Override
//    public void run() {
//        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
//        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
//        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
//        final Tensor outputTensor = outputTuple[0].toTensor();
//        final float[] outputs = outputTensor.getDataAsFloatArray();
//        final ArrayList<Result> results = PrePostProcessor.outputsToNMSPredictions(outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
//
//        runOnUiThread(() -> {
//            mButtonDetect.setEnabled(true);
//            mButtonDetect.setText(getString(R.string.detect));
//            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
//            mResultView.setResults(results);
//            mResultView.invalidate();
//            mResultView.setVisibility(View.VISIBLE);
//        });
//    }

    //    private static final int DOG = 12;
//    private static final int PERSON = 15;
//    private static final int SHEEP = 17;
    @Override
    public void run() {
//        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
//                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
//        final float[] inputs = inputTensor.getDataAsFloatArray(); // TODO: this should be the on-disk LUT image converted to tensor form
//
        final long startTime = SystemClock.elapsedRealtime();
//
//        // TODO: REST API inference would also be performed here via the served image at the endpoint
//        // TODO: update here for the associated demo input image to output LUT
//        Map<String, IValue> outTensors = (Map<String, IValue>) inputTensor;
//        outTensors =

        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("ImageSegmentation", "inference time (ms): " + inferenceTime);

//        final Tensor outputTensor = outTensors.get("out").toTensor();
        final Tensor outputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] scores = outputTensor.getDataAsFloatArray();
        int width = mBitmap.getWidth();
        int height = mBitmap.getHeight();
        Log.d(TAG, String.valueOf(mBitmap.getColorSpace()));
        Log.d(TAG, String.format("LUT Height: %d Width: %d", height, width));
        int[] intValues = new int[width * height];

        // apply coloring
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {

//                int maxi = 0, maxj = 0, maxk = 0;
//                double maxnum = -Integer.MAX_VALUE;
//                double maxnum = -Double.MAX_VALUE;

                for (int i = 0; i < CLASSNUM; i++) {
                    Log.d(TAG, String.format("Pixel Value: %d", scores[j * k + i]));

                }

//                    float score = scores[i * (width * height) + j * width + k];
//                    if (score > maxnum) {
//                        maxnum = score;
//                        maxi = i;
//                        maxj = j;
//                        maxk = k;
//                    }
//                    if (maxi == WATER)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                    else if (maxi == DOUBLE_PLANT)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                    else if (maxi == PLANTER_SKIP)
//                        intValues[maxj * width ] = COLOR_MAP[maxi];
//                    else if (maxi == DRYDOWN)
//                        intValues[maxj * width ] = COLOR_MAP[maxi];
//                    else if (maxi == WATERWAY)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                    else if (maxi == WEED_CLUSTER)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                    else if (maxi == ENDROW)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                    else if (maxi == NUTRIENT_DEFICIENT)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                    else if (maxi == STORM_DAMAGE)
//                        intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                }

                // color code is in HEX ie 2-leftmost bits define MAX range
                // FF = 255
                // remainig 6 bits define the color
                // useful resource: https://www.rapidtables.com/convert/color/hex-to-rgb.html
                // note that each color is mapped to a particualr label recall LUT
                // the "mask" provided when we perofrm inference yields an output image creates
                // this feature map

                // interpret tensor as a 1D array where the  maxing pixel
                // index will be mapped to the particular color code
//                if (maxi == WATER)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == DOUBLE_PLANT)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == PLANTER_SKIP)
//                    intValues[maxj * width ] = COLOR_MAP[maxi];
//                else if (maxi == DRYDOWN)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == WATERWAY)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == WEED_CLUSTER)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == ENDROW)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == NUTRIENT_DEFICIENT)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
//                else if (maxi == STORM_DAMAGE)
//                    intValues[maxj * width + maxk] = COLOR_MAP[maxi];
            }
        }

        Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
        Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
        outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
        final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mImageView.setImageBitmap(transferredBitmap);
//                mButtonSegment.setEnabled(true);
//                mButtonSegment.setText(getString(R.string.segment));
//                mProgressBar.setVisibility(ProgressBar.INVISIBLE);

            }
        });
    }

    public static Bitmap createTransparentBitmapFromBitmap(Bitmap bitmap,
                                                           int replaceThisColor) {
        if (bitmap != null) {
            int picw = bitmap.getWidth();
            int pich = bitmap.getHeight();
            int[] pix = new int[picw * pich];
            bitmap.getPixels(pix, 0, picw, 0, 0, picw, pich);

            for (int y = 0; y < pich; y++) {
                // from left to right
                for (int x = 0; x < picw; x++) {
                    int index = y * picw + x;
                    int r = (pix[index] >> 16) & 0xff;
                    int g = (pix[index] >> 8) & 0xff;
                    int b = pix[index] & 0xff;

                    if (pix[index] == replaceThisColor) {
                        pix[index] = Color.TRANSPARENT;
                    } else {
                        break;
                    }
                }

                // from right to left
                for (int x = picw - 1; x >= 0; x--) {
                    int index = y * picw + x;
                    int r = (pix[index] >> 16) & 0xff;
                    int g = (pix[index] >> 8) & 0xff;
                    int b = pix[index] & 0xff;

                    if (pix[index] == replaceThisColor) {
                        pix[index] = Color.TRANSPARENT;
                    } else {
                        break;
                    }
                }
            }

            Bitmap bm = Bitmap.createBitmap(pix, picw, pich,
                    Bitmap.Config.ARGB_4444);

            return bm;
        }
        return null;
    }

    public static Bitmap overlay(Bitmap bmp1, Bitmap bmp2) {
        Bitmap bmOverlay = Bitmap.createBitmap(bmp1.getWidth(), bmp1.getHeight(), bmp1.getConfig());
        Canvas canvas = new Canvas(bmOverlay);
        canvas.drawBitmap(bmp1, new Matrix(), null);
        canvas.drawBitmap(bmp2, 0, 0, null);
        bmp1.recycle();
        bmp2.recycle();
        return bmOverlay;
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        // base call to handle for CPU,GPU
//        super.onItemSelected(parent, view, pos, id);
        // handle for update
        if (parent == modeSpinner) {
            String mode = String.valueOf(parent.getItemAtPosition(pos).toString());
            if (mode.equals("transparent-overlay")) {
                overlayMode = 1;
            } else if (mode.equals("non-transparent")) {
                overlayMode = 0;
            }
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {
        // Do nothing
    }

}
