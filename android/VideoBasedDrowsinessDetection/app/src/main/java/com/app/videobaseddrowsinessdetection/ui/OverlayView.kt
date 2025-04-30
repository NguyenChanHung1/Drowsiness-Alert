package com.app.videobaseddrowsinessdetection.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView (context: Context, attrs: AttributeSet?=null) : View(context, attrs){
    private var box: RectF? = null
    private var pointX: Float? = null
    private var pointY: Float? = null
    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    fun updateBox(xCenter: Float, yCenter: Float, width: Float, height: Float) {
        val left = xCenter - width / 2
        val top = yCenter - height / 2
        val right = xCenter + width / 2
        val bottom = yCenter + height / 2
        pointX = xCenter
        pointY = yCenter
        box = RectF(left, bottom, right, top)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        box?.let {
            canvas.drawRect(it, paint)
        }
        if (pointX != null && pointY != null) {
            canvas.drawCircle(pointX!!, pointY!!, 2f, paint)
        }
    }
}