package com.app.videobaseddrowsinessdetection.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter.Options
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.gpu.GpuDelegate
import androidx.core.graphics.scale
import androidx.core.graphics.get

class YOLOInterpreter(context: Context) {
    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate ?= null
    private lateinit var boxes: ArrayList<Float>

    init {
        val model = FileUtil.loadMappedFile(context, "yolo8n_face_fp16_imgsz384.tflite")
        val options = Options()
        options.setNumThreads(8)
        interpreter = Interpreter(model, options)
    }

    fun predict(bitmap: Bitmap): ArrayList<Float> {
        val modelInputSize = 384 // 384x384x3
        val resizedBitmap = bitmap.scale(modelInputSize, modelInputSize)
        // input: 1x384x384x3
        val input = Array(1) { Array(modelInputSize) { Array(modelInputSize) { FloatArray(3) } } }
        for (y in 0 until modelInputSize) {
            for (x in 0 until modelInputSize) {
                val pixel = resizedBitmap[y,x]
                input[0][y][x][0] = (pixel shr 16 and 0xFF) / 255f
                input[0][y][x][1] = (pixel shr 8 and 0xFF) / 255f
                input[0][y][x][2] = (pixel and 0xFF) / 255f
            }
        }
        val output = Array(1) { Array(5) { FloatArray(3024) } }
        interpreter.run(input, output)

        boxes = arrayListOf<Float>(output[0][4][0], output[0][0][0], output[0][1][0], output[0][2][0],
            output[0][3][0])
        var max_thres = 0.0f
        var idx_max_thres = 0
//        for (i in 0 until 3024) {
//            val conf = output[0][4][i]
//            if (conf > 0.3 && conf > max_thres) {
//                max_thres = conf
//                idx_max_thres = i
//            }
//            if (conf > 0.3) {  // confidence threshold
//
////                boxes.add(RectF(left, top, right, bottom))
//            }
//        }

        // Return the first and the only value of the output vector
        val x = output[0][0][idx_max_thres]
        val y = output[0][1][idx_max_thres]
        val w = output[0][2][idx_max_thres]
        val h = output[0][3][idx_max_thres]
//        val x = (x2-x1) / 2
//        val y = (y2-y1) / 2
//        val w = x2 - x1
//        val h = y2 - y1
        val left = x - w / 2
        val top = y - h / 2
        val right = x + w / 2
        val bottom = y + h / 2
        boxes = arrayListOf<Float>(max_thres,1-x,1-y,w,h)

        return boxes
    }

    private fun preprocess(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        // Placeholder preprocessing
        return Array(1) { Array(112) { Array(112) { FloatArray(3) } } }
    }
}