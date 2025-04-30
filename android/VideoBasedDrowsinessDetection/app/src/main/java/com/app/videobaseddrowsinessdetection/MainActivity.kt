package com.app.videobaseddrowsinessdetection

import android.content.pm.PackageManager
import android.content.res.Resources
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.view.ViewCompat
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.camera.core.ImageProxy
import com.app.videobaseddrowsinessdetection.ml.YOLOInterpreter
import java.io.ByteArrayOutputStream
import androidx.core.graphics.scale
import com.app.videobaseddrowsinessdetection.ui.OverlayView

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var textView: TextView
    private lateinit var textViewW: TextView
    private lateinit var textViewH: TextView
    private lateinit var textViewXY: TextView
    private lateinit var progressButton: Button
    private lateinit var yoloModel: YOLOInterpreter
    private lateinit var overlayView: OverlayView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            yoloModel = YOLOInterpreter(applicationContext)
            Log.w("YOLO", "Works fine")
        }
        catch (e : Exception) {
            Toast.makeText(this, "Failed", Toast.LENGTH_LONG).show()
            Log.e("YOLO", e.toString())
        }
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), 1001)
        } else {
            startCamera()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        textView = findViewById<TextView>(R.id.textView)
        textViewH = findViewById<TextView>(R.id.textViewHeight)
        textViewW = findViewById<TextView>(R.id.textViewWidth)
        textViewXY = findViewById<TextView>(R.id.textViewX)
        textView.setText("Detecting...")
        overlayView = findViewById<OverlayView>(R.id.overlayView)
        progressButton = findViewById<Button>(R.id.showProgressBtn)

        Log.w("YOLO", "camera detected")
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
                }
            val imageAnalyzer = ImageAnalysis.Builder().build().also {
                it.setAnalyzer(ContextCompat.getMainExecutor(this), { imageProxy ->

                    // Placeholder: convert imageProxy to bitmap and run inference
                    val bitmap = imageProxyToBitmap(imageProxy)
                    val result = yoloModel.predict(bitmap) // FloatArray
                    Log.d("YOLO", "imageProxy -> Bitmap: done")


                    val viewW = overlayView.width.toFloat()
                    val viewH = overlayView.height.toFloat()
                    var x = 479.0f//0.0f
                    var y = 674.5f//0.0f
                    var w = 958.0f//0.0f
                    var h = 1349.0f//0.0f
                    if(result.get(0) > 0.3) {
                        x = (result.get(1)) * viewW
                        y = (result.get(2)) * viewH
                        w = (result.get(3)) * viewW
                        h = (result.get(4)) * viewH
                    }
                    runOnUiThread {
                        overlayView.updateBox(x,y,w,h)
                    }
//                    runOnUiThread {
//                        val centerX = overlayView.width / 2f
//                        val centerY = overlayView.height / 2f
//                        val boxW = 100f
//                        val boxH = 150f
//                        overlayView.updateBox(centerX, centerY, boxW, boxH)
//                    }

                    textView.setText("Confidence? = " + result.get(0))
                    textViewW.setText("Width? = " +result.get(3)+"/"+viewW)
                    textViewH.setText("Height? = " + result.get(4)+"/"+viewH)
                    textViewXY.setText("XY Center? = " + result.get(1) + ", " + result.get(2))

                    Log.w("Hehe", "Set text")
                    imageProxy.close()
                })
            }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val yuv = out.toByteArray()
        return BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                android.Manifest.permission.CAMERA
            ).apply {
//                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
//                    add(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
//                }
            }.toTypedArray()
    }
}