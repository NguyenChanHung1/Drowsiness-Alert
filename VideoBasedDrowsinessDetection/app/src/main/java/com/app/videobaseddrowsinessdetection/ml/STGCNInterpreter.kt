package com.app.videobaseddrowsinessdetection.ml

import android.content.ContentValues.TAG
import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Interpreter.Options
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.util.concurrent.LinkedBlockingDeque

class STGCNInterpreter (context: Context) {
    private val interpreter: Interpreter
    private lateinit var gpuDelegate: GpuDelegate
    private var inputScale = 1f
    private var inputZeroPoint = 0
    private var outputScale = 1f
    private var outputZeroPoint = 0

    init {
        val options = Options()
        val compatList = CompatibilityList()

        if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
            } catch (e: Exception) {

            }
        } else {
            try{
                val nnApiOptions = NnApiDelegate.Options().apply {
                    setUseNnapiCpu(true)
                    setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                }
                options.addDelegate(NnApiDelegate(nnApiOptions))
            }
            catch (e: Exception) {
                options.setUseXNNPACK(true)
            }

            options.setNumThreads(4)
        }

        val fileDescriptor: AssetFileDescriptor = context.assets.openFd("best_model_epoch_test_t40_twostream_uint8.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val byteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(byteBuffer, options)

        // Read quantization params
        val inputTensor = interpreter.getInputTensor(0)
        val inputQuantParams = inputTensor.quantizationParams()
        inputScale = inputQuantParams.scale
        inputZeroPoint = inputQuantParams.zeroPoint

        val outputTensor = interpreter.getOutputTensor(0)
        val outputQuantParams = outputTensor.quantizationParams()
        outputScale = outputQuantParams.scale
        outputZeroPoint = outputQuantParams.zeroPoint
    }

    fun predict(listOfLandmarks: LinkedBlockingDeque<MutableList<Pair<Float, Float>>>) : ArrayList<Float>{
        val input = Array(1) { Array(2) { Array(40) { ByteArray(30) } } }
        val landmarksList = listOfLandmarks.toList().takeLast(40)
        for (i in 0 until 40) {
            val frame_i_landmarks = landmarksList[i] // MutableList<Pair<Float, Float>>

            for (j in 0 until 30) {
                val x = frame_i_landmarks[j].first
                val y = frame_i_landmarks[j].second

                val quantizedX = ((x / inputScale) + inputZeroPoint).toInt().coerceIn(0, 255)
                val quantizedY = ((y / inputScale) + inputZeroPoint).toInt().coerceIn(0, 255)

                input[0][0][i][j] = quantizedX.toByte()
                input[0][1][i][j] = quantizedY.toByte()
            }
        }

        val output = Array(1) { ByteArray(2) }
        interpreter.run(input, output)

        val result = output[0].map {
            ((it.toInt() and 0xFF) - outputZeroPoint) * outputScale
        }

        return ArrayList(result)
    }
}