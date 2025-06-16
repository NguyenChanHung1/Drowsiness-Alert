package com.app.videobaseddrowsinessdetection.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.core.graphics.scale
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class BitmapUtils {

    companion object {
        private var ORG_IMG_X: Float = 0.0f
        private var ORG_IMG_Y: Float = 0.0f
        private var FACE_IMG_X: Float = 0.0f
        private var FACE_IMG_Y: Float = 0.0f
        private var PRED_ORG_IMG_X: Float = 0.0f
        private var PRED_ORG_IMG_Y: Float = 0.0f
        private var PRED_FACE_X: Float = 0.0f
        private var PRED_FACE_Y: Float = 0.0f

        fun rotateAndMirrorNV21(input: ByteArray, width: Int, height: Int): ByteArray {
            val frameSize = width * height
            val output = ByteArray(input.size)

            val rotatedWidth = height
            val rotatedHeight = width

            // === Rotate Y plane 270° + flip horizontally ===
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val srcIdx = y * width + x

                    // Rotate 270° clockwise: (x, y) -> (y, width - x - 1)
                    val rotatedX = y
                    val rotatedY = width - x - 1

                    // Flip horizontally after rotation: flip x
                    val flippedX = rotatedWidth - rotatedX - 1
                    val dstIdx = rotatedY * rotatedWidth + flippedX

                    output[dstIdx] = input[srcIdx]
                }
            }

            val uvWidth = width / 2
            val uvHeight = height / 2

            for (y in 0 until uvHeight) {
                for (x in 0 until uvWidth) {
                    val srcIdx = frameSize + y * width + x * 2
                    val v = input[srcIdx]
                    val u = input[srcIdx + 1]

                    // Rotate 270°: (x, y) -> (y, uvWidth - x - 1)
                    val rotatedX = y
                    val rotatedY = uvWidth - x - 1

                    // Flip horizontally after rotation
                    val flippedX = rotatedWidth / 2 - rotatedX - 1
                    val dstIdx = frameSize + rotatedY * rotatedWidth + flippedX * 2

                    output[dstIdx] = v
                    output[dstIdx + 1] = u
                }
            }

            return output
        }

        fun imageProxyToYuv(image: ImageProxy): ByteArray {
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

            val new_nv21 = rotateAndMirrorNV21(nv21, image.width, image.height)

            val yuvImage = YuvImage(new_nv21, ImageFormat.NV21, image.height, image.width, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.height, image.width), 100, out)
            val yuv = out.toByteArray()
//            return BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
            return yuv
        }

        fun imageProxyToNV21(image: ImageProxy): ByteArray {
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

            val new_nv21 = rotateAndMirrorNV21(nv21, image.width, image.height)

            val yuvImage = YuvImage(new_nv21, ImageFormat.NV21, image.height, image.width, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.height, image.width), 100, out)
            val yuv = out.toByteArray()
//            return BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
            return new_nv21
        }

        fun cropNV21(nv21: ByteArray, width: Int, height: Int, rect: Rect): ByteArray {
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val outputStream = ByteArrayOutputStream()
            val success = yuvImage.compressToJpeg(rect, 100, outputStream)
            val jpegData = (outputStream.toByteArray()) // This can be decoded or fed directly
//            return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
            return jpegData
        }

        fun createFaceBitmap(
            orgBitmap: Bitmap,
            xCenter: Float, yCenter: Float, width: Float, height: Float
        ) : Bitmap {
            // Rotate by -90 degrees
            val matrix = Matrix()
            matrix.postRotate(0f)
//            matrix.preScale(-1.0f,1.0f)

            var intX = (xCenter - width / 2).toInt()
            var intY = (yCenter - height / 2).toInt()
            var intW = width.toInt()
            var intH = height.toInt()

            // Clamp to bitmap bounds
            intX = intX.coerceAtLeast(0)
            intY = intY.coerceAtLeast(0)
            if (intX + intW > orgBitmap.width) intW = orgBitmap.width - intX
            if (intY + intH > orgBitmap.height) intH = orgBitmap.height - intY

            val faceBitmap = Bitmap.createBitmap(orgBitmap, intX, intY, intW, intH, matrix, true)
            return faceBitmap
        }

        fun interpretBox(box: ArrayList<Float>, viewH: Float, viewW: Float, display: Boolean=true) : ArrayList<Float> {
            if (box.size < 5) return arrayListOf(0f, 0f, 0f, 1f, 1f)  // return a default box

            var x = 0.0f
            var y = 0.0f
            var w = 0.0f
            var h = 0.0f
            if(box.get(0) > 0.5) {
                x = (box.get(1))
                y = (box.get(2))
                w = (box.get(3))
                h = (box.get(4))
            }
            if(viewW != 0f && viewH != 0f) {
                x = (x) * viewW ; y = (y) * viewH;  w = (w) * viewW; h = h * viewH
                // Save variables for PFLD result interpreting
                if(display) {
                    ORG_IMG_X = x - w / 2
                    ORG_IMG_Y = y - h / 2
                    FACE_IMG_X = w
                    FACE_IMG_Y = h
                }
                else {
                    PRED_ORG_IMG_X = x - w / 2
                    PRED_ORG_IMG_Y = y - h / 2
                    PRED_FACE_X = w
                    PRED_FACE_Y = h
                }
            }
            else {
                x = 0f; y = 0f; w = 1f; h = 1f
            }

            return arrayListOf<Float>(box.get(0),x,y,w,h)
        }

        fun interpretLandmarks(landmarks: MutableList<Pair<Float, Float>>,
                               org_width: Float,
                               org_height: Float,
                               display: Boolean=true) : MutableList<Pair<Float, Float>> {
            val listLandmarks = mutableListOf<Pair<Float, Float>>()
            for ((x, y) in landmarks) {
                if(display) {
                    val dispX = x * FACE_IMG_X + ORG_IMG_X
                    val dispY = y * FACE_IMG_Y + ORG_IMG_Y
                    listLandmarks.add(Pair(dispX, dispY))
                }
                else {
                    val dispX = (x * PRED_FACE_X + PRED_ORG_IMG_X) / org_width
                    val dispY = (y * PRED_FACE_Y + PRED_ORG_IMG_Y) / org_height
                    listLandmarks.add(Pair(dispX, dispY))
                }
            }

            return listLandmarks
        }
    }
}