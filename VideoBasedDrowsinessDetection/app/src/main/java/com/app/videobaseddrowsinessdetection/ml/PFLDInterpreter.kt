package com.app.videobaseddrowsinessdetection.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Environment
import android.util.Log
import org.tensorflow.lite.Interpreter.Options
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import androidx.core.graphics.scale
import androidx.core.graphics.get
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class PFLDInterpreter(context: Context) {
    private val interpreter: Interpreter
    // perform prediction uint8
    private var inputScale = 1.0f
    private var inputZeroPoint = 0
    private var outputScale = 1.0f
    private var outputZeroPoint = 0
    private val SELECTED_POINTS = arrayListOf(0, 2, 5, 8, 11, 14, 16, 36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 27, 29, 31, 33, 35, 60, 61, 63, 64, 65, 67)

    init {
        val model = FileUtil.loadMappedFile(context, "pfld_int8_250ep_final.tflite")
        val options = Options()
        options.setUseXNNPACK(true)
        options.setNumThreads(4)
        interpreter = Interpreter(model, options)

        // uint8
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

    fun predict(yuvData: ByteArray): MutableList<Pair<Float, Float>> {
        val landmarks = mutableListOf<Pair<Float, Float>>()// 68 keypoints
        val bitmap = BitmapFactory.decodeByteArray(yuvData, 0, yuvData.size)

        val modelInputSize = 112 // 112x112x3
        val resizedBitmap = bitmap.scale(modelInputSize, modelInputSize)
//        saveBitmapToStorage(resizedBitmap)

        // Prepare input ByteBuffer: [1, 112, 112, 3], dtype=UINT8
        val inputBuffer = Array(1) { Array(3) { Array(modelInputSize) { ByteArray(modelInputSize) } } }
//        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until modelInputSize) {
            for (x in 0 until modelInputSize) {
                val pixel = resizedBitmap[x, y]

                inputBuffer[0][0][y][x] = (((pixel shr 16 and 0xFF) / 255f) / inputScale
                + inputZeroPoint).toInt().coerceIn(0,255).toByte()
                inputBuffer[0][1][y][x] = (((pixel shr 8 and 0xFF) / 255f) / inputScale
                        + inputZeroPoint).toInt().coerceIn(0,255).toByte()
                inputBuffer[0][2][y][x] = (((pixel and 0xFF) / 255f) / inputScale
                        + inputZeroPoint).toInt().coerceIn(0,255).toByte()
            }
        }

        // Output is 1 x 136 (if still quantized: uint8)
        val outputArray = Array(1) { ByteArray(136) }
        interpreter.run(inputBuffer, outputArray)

        // Avoid stacking up values
        landmarks.clear()
        for (i in 0 until 68) {
            if(i in SELECTED_POINTS) {
                val x = (outputArray[0][i * 2].toUByte().toInt() - outputZeroPoint) * outputScale
                val y =
                    (outputArray[0][i * 2 + 1].toUByte().toInt() - outputZeroPoint) * outputScale
                landmarks.add(Pair(x, y))
            }
            else continue
        }

        return landmarks
    }
}