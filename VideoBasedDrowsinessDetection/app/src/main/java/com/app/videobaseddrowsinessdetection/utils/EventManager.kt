package com.app.videobaseddrowsinessdetection.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.text.SimpleDateFormat
import java.util.*

class EventManager(private val context: Context) {
    private val sharedPreferences: SharedPreferences = context.getSharedPreferences(
        EVENT_PREF_NAME, Context.MODE_PRIVATE
    )
    private val gson = Gson()

    // Record event with timestamp
    fun recordEvent(startTime: Long, type: Int, duration: Float) {
        try {
            val events = getAllEvents().toMutableList()
//            val timestamp = System.currentTimeMillis()
            val newEvent = Event(type, startTime, duration)
            events.add(newEvent)

            // Save updated list
            val jsonEvents = gson.toJson(events)
            sharedPreferences.edit().putString(EVENTS_KEY, jsonEvents).apply()

            Log.i(TAG, "Event recorded at ${newEvent.getFormattedTime()}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to record drowsiness event: ${e.message}")
        }
    }

    // Retrieve all drowsiness event -> list of DrowsinessEvent objects
    fun getAllEvents(): List<Event> {
        val jsonEvents = sharedPreferences.getString(EVENTS_KEY, null) ?: return emptyList()

        try {
            val type = object : TypeToken<List<Event>>() {}.type // type = List<Event>
            val events: List<Event> = gson.fromJson(jsonEvents, type)
            return events.sortedByDescending { it.timestamp }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to retrieve drowsiness events: ${e.message}")
            return emptyList()
        }
    }

    fun clearAllEvents() {
        sharedPreferences.edit().remove(EVENTS_KEY).apply()
        Log.i(TAG, "All drowsiness events cleared")
    }

    fun getTotalEvents(): Int {
        return getAllEvents().size
    }

    companion object {
        private const val TAG = "DrowsinessEventManager"
        private const val EVENT_PREF_NAME = "event_prefs"
        const val EVENTS_KEY = "events"
        const val TYPE_DROWSY = 0
        const val TYPE_NOFACE = 1
    }
}

// class representing Event
class Event(val type: Int, val timestamp: Long, val duration: Float) {

    fun getFormattedTime(): String {
        val dateFormat = SimpleDateFormat("MMM dd, yyyy 'at' hh:mm:ss a", Locale.getDefault())
        return dateFormat.format(Date(timestamp))
    }

    fun getDurationText(): String {
        return String.format("%.1f",this.duration) + "s"
    }

    fun getType() : String {
        var typeString = ""
        when (this.type) {
            EventManager.TYPE_DROWSY -> typeString += "Drowsy"
            EventManager.TYPE_NOFACE -> typeString += "No face detected"
        }
        return typeString
    }
}