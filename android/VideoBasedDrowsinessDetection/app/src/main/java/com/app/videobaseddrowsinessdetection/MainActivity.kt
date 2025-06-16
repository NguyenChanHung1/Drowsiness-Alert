package com.app.videobaseddrowsinessdetection

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
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
import android.view.Display
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Toast
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import java.io.ByteArrayOutputStream
import androidx.core.app.NotificationCompat
import com.app.videobaseddrowsinessdetection.ml.MLPipeline
import com.app.videobaseddrowsinessdetection.ml.PFLDInterpreter
import com.app.videobaseddrowsinessdetection.ml.STGCNInterpreter
import com.app.videobaseddrowsinessdetection.ui.OverlayView
import com.app.videobaseddrowsinessdetection.utils.BitmapUtils
import com.app.videobaseddrowsinessdetection.utils.EventManager
import com.app.videobaseddrowsinessdetection.utils.FaceDetectionListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.time.measureTime

class MainActivity : AppCompatActivity(), FaceDetectionListener {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var textView: TextView
    private lateinit var mlPipe: MLPipeline
    private lateinit var progressButton: Button
    private lateinit var overlayView: OverlayView
    private lateinit var previewView: PreviewView
    private lateinit var btnStartCamera: Button
    private lateinit var cameraPlaceholder: LinearLayout
    private lateinit var btnShowHistory: Button
    private lateinit var statusIcon: ImageView
    private lateinit var controller: LifecycleCameraController
    private lateinit var eventManager: EventManager

    // Drowsiness tracking
    private var mediaPlayer: MediaPlayer? = null
    private var drowsyStartTime = 0L // millisecond
    private var isDrowsy = false
    private var nofaceStartTime = 0L
    private var faceDetected = false
    private var notificationSent = false

    val sequenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mlPipe = MLPipeline(this)

        // Run on a coroutine
        var init_status = 0
        GlobalScope.launch(Dispatchers.IO) {
            init_status = coroutineScope {
                val status = async {
                    mlPipe.initialize(applicationContext)
                }
                status.await()
            }
//            mlPipe.initialize(applicationContext)
        }

        if(init_status == 1) {
            Toast.makeText(this, "Failed to initialize models", Toast.LENGTH_LONG).show()
        }
        mlPipe.initialize(applicationContext)

        eventManager = EventManager(applicationContext)

        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        previewView = findViewById<PreviewView>(R.id.previewView)
        overlayView = findViewById<OverlayView>(R.id.overlayView)
        progressButton = findViewById<Button>(R.id.showProgressBtn)
        btnStartCamera = findViewById<Button>(R.id.startCameraBtn)
        textView = findViewById<TextView>(R.id.textView)
        btnShowHistory = findViewById<Button>(R.id.showHistoryBtn)
        cameraPlaceholder = findViewById<LinearLayout>(R.id.camera_placeholder)
        statusIcon = findViewById(R.id.status_icon)

        // add listener
        progressButton.setOnClickListener {
            if (overlayView.visibility == View.GONE) {
                overlayView.visibility = View.VISIBLE
                progressButton.setText("Hide face landmarks")
            }
            else {
                overlayView.visibility = View.GONE
                progressButton.setText("Show face landmarks")
            }
        }

        btnShowHistory.setOnClickListener {
            val intent = Intent(this, HistoryActivity::class.java)
            startActivity(intent)
        }

        //
        controller = LifecycleCameraController(applicationContext).apply {
                setEnabledUseCases(
                    CameraController.IMAGE_CAPTURE or CameraController.VIDEO_CAPTURE
                )
            }

        btnStartCamera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) {
                var permissionToRequest = arrayOf(android.Manifest.permission.CAMERA,
                    android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    android.Manifest.permission.READ_EXTERNAL_STORAGE)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                    ContextCompat.checkSelfPermission(this, android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                    permissionToRequest += arrayOf(android.Manifest.permission.POST_NOTIFICATIONS)
                }
                ActivityCompat.requestPermissions(
                    this,
                    permissionToRequest,
                    1001
                )
            } else {
                startCamera()
                previewView.visibility = View.VISIBLE
                overlayView.visibility = View.GONE // by default hide landmarks
                progressButton.visibility = View.VISIBLE
                textView.visibility = View.VISIBLE
                btnStartCamera.visibility = View.GONE
                findViewById<LinearLayout>(R.id.linearlayoutTextView).visibility = View.VISIBLE
                cameraPlaceholder.visibility = View.GONE
            }
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        textView.setText("Detecting...")
        overlayView = findViewById<OverlayView>(R.id.overlayView)
        progressButton = findViewById<Button>(R.id.showProgressBtn)

        updateEventCountDisplay()
        startSequenceConsumer()
    }

    private fun showDrowsinessNotification(durationSeconds: Float) {
        val notificationText = "Drowsiness detected for ${"%.1f".format(durationSeconds)} seconds!"
        Toast.makeText(this, notificationText, Toast.LENGTH_LONG).show()

        if (mediaPlayer == null) {
            mediaPlayer = MediaPlayer.create(this, R.raw.alarm_clock)
            mediaPlayer?.isLooping = true
        }

        if (mediaPlayer?.isPlaying == false) {
            mediaPlayer?.start()
        }
    }

    override fun onNoFaceDetected(timestamp: Long, detected: Boolean) {
        if(faceDetected && !detected) {
            nofaceStartTime = timestamp
            faceDetected = false
        }
        else if(!faceDetected && detected) {
            val duration = (timestamp - nofaceStartTime) / 1000f
            if(duration > 5.0f) {
                eventManager.recordEvent(nofaceStartTime, EventManager.TYPE_NOFACE, duration)
            }
            faceDetected = true
        }
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
                    mlPipe.processFrame(
                        imageProxy, overlayView
                    )
//                    imageProxy.close()
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

    private fun startSequenceConsumer() {
        sequenceScope.launch {
            while (isActive) {
                val timeSeq = measureTime {
                    val flag = mlPipe.processSequence()

                    withContext(Dispatchers.Main) {
                        when (flag) {
                            -1 -> {
                                textView.text = "Detecting ..."
                                statusIcon.setImageResource(R.drawable.ic_status_detecting)
                                notificationSent = false
                            }

                            0 -> {
                                textView.text = "Status: Focused"
                                statusIcon.setImageResource(R.drawable.ic_status_focused)

                                // if previously drowsy, calc duration
                                if(isDrowsy && drowsyStartTime > 0) {
                                    val duration = (System.currentTimeMillis() - drowsyStartTime) / 1000f
                                    if(duration > 10.0f){
                                        withContext(Dispatchers.IO) {
                                            eventManager.recordEvent(drowsyStartTime,
                                                EventManager.TYPE_DROWSY,
                                                duration)
                                        }
                                    }
                                }
                                isDrowsy = false
                                drowsyStartTime = 0L
                                notificationSent = false
                                mediaPlayer?.pause()
                                mediaPlayer?.seekTo(0)
                            }

                            1 -> {
                                textView.text = "Status: Drowsy"
                                statusIcon.setImageResource(R.drawable.ic_status_drowsy)
                                if(!isDrowsy) {
                                    isDrowsy = true
                                    drowsyStartTime = System.currentTimeMillis()
                                    notificationSent = false
                                }
                                val byfar_duration = (System.currentTimeMillis() - drowsyStartTime) / 1000f
                                if( byfar_duration >= 5f && !notificationSent) {
                                    showDrowsinessNotification(byfar_duration)
                                    notificationSent = true
                                }
                            }
                        }
                        updateEventCountDisplay()
                    }
                    Log.d("Sequence time", "flag hehe flag=${flag}")
                }
                Log.d("Sequence time", "time sequence ${timeSeq}, ${textView.text}, thread ${Thread.currentThread().name}")
//                delay(100) // avoid busy loop; tune this value as needed
            }
        }
    }

    private fun updateEventCountDisplay() {
        val count = eventManager.getTotalEvents()
        btnShowHistory.text = "History ($count)"
    }

    companion object {
        private const val TAG = "DrowsyApp"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                android.Manifest.permission.CAMERA,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                android.Manifest.permission.READ_EXTERNAL_STORAGE
            ).toTypedArray()
    }
}