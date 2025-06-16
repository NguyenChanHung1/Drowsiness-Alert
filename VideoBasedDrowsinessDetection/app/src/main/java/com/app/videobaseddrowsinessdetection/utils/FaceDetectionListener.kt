package com.app.videobaseddrowsinessdetection.utils

interface FaceDetectionListener {
    fun onNoFaceDetected(timestamp: Long, detected: Boolean)
}