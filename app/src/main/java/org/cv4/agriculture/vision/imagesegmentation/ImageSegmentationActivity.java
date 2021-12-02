package org.cv4.agriculture.vision.imagesegmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;

import org.cv4.agriculture.vision.R;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;


public class ImageSegmentationActivity extends AppCompatActivity implements Runnable {
    private ImageView mImageView;
    private Button mButtonSegment;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private String mImagename = "deeplab.jpg";

    // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
    private static final int CLASSNUM = 21;
    private static final int DOG = 12;
    private static final int PERSON = 15;
    private static final int SHEEP = 17;

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
        setContentView(R.layout.activity_image_segmention);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);

        final Button buttonRestart = findViewById(R.id.restartButton);
        buttonRestart.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (mImagename == "deeplab.jpg")
                    mImagename = "dog.jpg";
                else
                    mImagename = "deeplab.jpg";
                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("ImageSegmentation", "Error reading assets", e);
                    finish();
                }
            }
        });


        mButtonSegment = findViewById(R.id.segmentButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonSegment.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));

                Thread thread = new Thread(ImageSegmentationActivity.this);
                thread.start();
            }
        });

        try {
//            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "rx50_optimized_scripted1.ptl"));
            mModule = LiteModuleLoader.load(ImageSegmentationActivity.assetFilePath(getApplicationContext(), "yolov5s.torchscript.ptl"));
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }
    }

    @Override
    public void run() {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final float[] inputs = inputTensor.getDataAsFloatArray();

        final long startTime = SystemClock.elapsedRealtime();
        Map<String, IValue> outTensors = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("ImageSegmentation",  "inference time (ms): " + inferenceTime);

        final Tensor outputTensor = outTensors.get("out").toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();
        int width = mBitmap.getWidth();
        int height = mBitmap.getHeight();
        int[] intValues = new int[width * height];
//        loop over the pixels
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {

                int maxi = 0, maxj = 0, maxk = 0;
                double maxnum = -Double.MAX_VALUE;



                for (int i = 0; i < CLASSNUM; i++) {
                    float score = scores[i * (width * height) + j * width + k];
                    if (score > maxnum) {
                        maxnum = score;
                        maxi = i; maxj = j; maxk = k;
                    }
                }

                // color code is in HEX ie 2-leftmost bits define MAX range
                // FF = 255
                // remainig 6 bits define the color
                // useful resource: https://www.rapidtables.com/convert/color/hex-to-rgb.html
                // note that each color is mapped to a particualr label recall LUT
                // the "mask" provided when we perofrm inference yields an output image creates
                // this feature map
                if (maxi == PERSON)
                    // interpret tensor as a 1D array where the  maxing pixel
                    // index will be mapped to the particular color code
                    intValues[maxj * width + maxk] = 0xFFFF0000;
                else if (maxi == DOG)
                    intValues[maxj * width + maxk] = 0xFF00FF00;
                else if (maxi == SHEEP)
                    intValues[maxj * width + maxk] = 0xFF0000FF;
                else
                    intValues[maxj * width + maxk] = 0xFF000000;
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
                mButtonSegment.setEnabled(true);

                mButtonSegment.setText(getString(R.string.segment));
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);

            }
        });
    }
}
