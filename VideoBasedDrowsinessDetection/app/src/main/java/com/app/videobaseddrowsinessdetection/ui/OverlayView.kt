package com.app.videobaseddrowsinessdetection.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View

class OverlayView (context: Context, attrs: AttributeSet?=null) : View(context, attrs){
    private var box: RectF = RectF(-0.5f,-0.5f,0.5f,0.5f)
    private var landmarks: MutableList<Pair<Float, Float>>? = null
    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    fun updateBox(xCenter: Float, yCenter: Float, width: Float, height: Float)  {
        val left = xCenter - width / 2
        val top = yCenter - height / 2
        val right = xCenter + width / 2
        val bottom = yCenter + height / 2
        box = RectF(left, top, right, bottom)
        Log.d("OverlayView", "${box.left}, ${box.top}, ${box.right}, ${box.bottom}")
        invalidate()
    }

    fun updateLandmarks(points: MutableList<Pair<Float, Float>>) {
        landmarks = points
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        box?.let {
            canvas.drawRect(it, paint)
        }
        landmarks?.let {
            for ((x,y) in landmarks) {
                canvas.drawCircle(x,y,2f,paint)
            }
        }
    }
}
