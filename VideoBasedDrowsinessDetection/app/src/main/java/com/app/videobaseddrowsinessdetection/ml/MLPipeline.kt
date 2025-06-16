package com.app.videobaseddrowsinessdetection.ml

import android.annotation.SuppressLint
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import com.app.videobaseddrowsinessdetection.ui.OverlayView
import com.app.videobaseddrowsinessdetection.utils.BitmapUtils
import com.app.videobaseddrowsinessdetection.utils.EventManager
import com.app.videobaseddrowsinessdetection.utils.FaceDetectionListener
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingDeque
import kotlin.math.max
import kotlin.math.min
import kotlin.time.measureTime

class MLPipeline(private val listener: FaceDetectionListener) {
//    private lateinit var detectModel: YuNetInterpreter
    private lateinit var detectModel: FaceDetector
    private lateinit var landmarkModel: PFLDInterpreter
    private lateinit var stgcnModel: STGCNInterpreter
    private val frameQueue: LinkedBlockingDeque<MutableList<Pair<Float, Float>>> = LinkedBlockingDeque(40)
    private lateinit var final_output: ArrayList<Float>
    private lateinit var context: Context
    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var drowsyFlag = -1
    private val modelLock = Mutex()
    @Volatile var isProcessingFrame = false // avoid critical resource conflict
    private var frameCounter = 0
    @Volatile var isProcessingSequence = false

    fun initialize(context: Context) : Int {
        try {
            this.context = context
            val highAccuracyOpts = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setMinFaceSize(0.5f)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build()
            detectModel = FaceDetection.getClient(highAccuracyOpts)
            landmarkModel = PFLDInterpreter(context)
            stgcnModel = STGCNInterpreter(context)
            return 0
        }
        catch (e : Exception) {
            return 1
        }
    }

    fun processFrame(imageProxy: ImageProxy, overlayView: OverlayView) {
//        val detectResult = detectModel.predict(bitmap) // FloatArray
        if(frameCounter>=2) {
            imageProxy.close()
            return
        }
        frameCounter++
        isProcessingFrame = true
        backgroundExecutor.execute {
            val bgetime = measureTime {
                val imageNV21 = BitmapUtils.imageProxyToNV21(imageProxy)
                val image = InputImage.fromByteArray(imageNV21, imageProxy.height, imageProxy.width,
                    0, InputImage.IMAGE_FORMAT_NV21)
                detectModel.process(image)
                    .addOnSuccessListener { faces ->
                        Log.d("DetectModel", "DetectModel thread: ${Thread.currentThread().name}")
                        if (faces.isNotEmpty()) {
                            // on main thread
                            val face = faces[0]
                            val bounds = face.boundingBox
                            val detectResult = ArrayList<Float>(5).apply {
                                add(1.0f)
                                val xCenter =
                                    (bounds.left.toFloat() + bounds.right.toFloat()) / 2f / image.width.toFloat()
                                val yCenter =
                                    (bounds.top.toFloat() + bounds.bottom.toFloat()) / 2f / image.height.toFloat()
                                val viewW = bounds.width() / image.width.toFloat()
                                val viewH = bounds.height() / image.height.toFloat()
                                addAll(arrayListOf(xCenter, yCenter, viewW, viewH))
                            }
                            backgroundExecutor.execute {
                                val timeBgExec = measureTime {
                                    val boxForDisplay = BitmapUtils.interpretBox(
                                        detectResult,
                                        overlayView.height.toFloat(),
                                        overlayView.width.toFloat(),
                                        true
                                    )
                                    val boxForPredict = BitmapUtils.interpretBox(
                                        detectResult,
                                        image.height.toFloat(),
                                        image.width.toFloat(),
                                        false
                                    )

                                    val left = max(0f, boxForPredict[1] - boxForPredict[3] / 2f)
                                    val top = max(0f, boxForPredict[2] - boxForPredict[4] / 2f)
                                    val right = min(image.width.toFloat(), boxForPredict[1] + boxForPredict[3] / 2f)
                                    val bottom = min(image.height.toFloat(), boxForPredict[2] + boxForPredict[4] / 2f)

                                    val faceNV21 = BitmapUtils.cropNV21(imageNV21, image.width,
                                        image.height, Rect(
                                            left.toInt(), top.toInt(), right.toInt(), bottom.toInt()
                                        ))

                                    val landmarksResult =
                                        landmarkModel.predict(faceNV21) // MutableList<Pair<Float,Float>>

                                    val landmarksForPredict =
                                        BitmapUtils.interpretLandmarks(
                                            landmarksResult, image.width.toFloat(),
                                            image.height.toFloat(), false
                                        )
                                    val landmarksForDisplay =
                                        BitmapUtils.interpretLandmarks(
                                            landmarksResult, image.width.toFloat(),
                                            image.height.toFloat(), true
                                        )

                                    synchronized(frameQueue) {
                                        if (!frameQueue.offerLast(landmarksForPredict)) {
                                            frameQueue.pollFirst()
                                            frameQueue.offerLast(landmarksForPredict)
                                        }
                                        Log.d("Queue", "added to queue")
                                    }

                                    // Update the overlayView asap
                                    mainHandler.post {
                                        overlayView.updateBox(
                                            boxForDisplay.get(1), boxForDisplay.get(2),
                                            boxForDisplay.get(3), boxForDisplay.get(4)
                                        )
                                        overlayView.updateLandmarks(landmarksForDisplay)
                                        listener.onNoFaceDetected(System.currentTimeMillis(), true)
                                    }
                                }
                            }
                        } else {
                            overlayView.updateBox(0f, 0f, 1f, 1f)
                            overlayView.updateLandmarks(
                                generateSequence { Pair(0f, 0f) }.take(30).toMutableList()
                            )
                            listener.onNoFaceDetected(System.currentTimeMillis(), false)
                        }
                    }
                    .addOnCompleteListener {
                        frameCounter--
                        imageProxy.close()
                    }
            }
            Log.d("DetectModel", "DetectModel thread: ${Thread.currentThread().name}, bge time: ${bgetime}")
        }
    }

    suspend fun processSequence() : Int {
        if(isProcessingSequence) {
            return this.drowsyFlag
        }
        isProcessingSequence = true

        if(frameQueue.size == 40) {
            modelLock.withLock {
                final_output = stgcnModel.predict(frameQueue)
            }
        }
        else {
            isProcessingSequence = false
            return this.drowsyFlag
        }

        Log.d("Sequence", "final output ${final_output[0]}, ${final_output[1]}")

        var isDrowsy = false
        if ((final_output[0] >= final_output[1]) || (final_output[0] < final_output[1] && final_output[1] <= 0.75)) {
            this.drowsyFlag = 0
            isDrowsy = false
        } else if (final_output[0] < final_output[1] ){//&& final_output[1] > 0.7) {
            this.drowsyFlag = 1

            if (!isDrowsy) {
                isDrowsy = true
            }
        } else {
            this.drowsyFlag = -1
        }
        isProcessingSequence = false

        return this.drowsyFlag
    }

}