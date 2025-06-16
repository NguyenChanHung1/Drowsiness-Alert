package com.app.videobaseddrowsinessdetection.adapters

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AnimationUtils
import android.widget.TextView
import androidx.cardview.widget.CardView
import androidx.recyclerview.widget.RecyclerView
import com.app.videobaseddrowsinessdetection.R
import com.app.videobaseddrowsinessdetection.utils.Event
import java.text.SimpleDateFormat
import java.util.Locale

class EventAdapter(private val events: List<Event>) :
    RecyclerView.Adapter<EventAdapter.ViewHolder>() {

    // Track the last position that was animated
    private var lastPosition = -1

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val cardView: CardView = view.findViewById(R.id.event_card)
        val timeText: TextView = view.findViewById(R.id.event_time)
        val dateText: TextView = view.findViewById(R.id.event_date)
        val durationText: TextView = view.findViewById(R.id.event_duration)
        val typeText: TextView = view.findViewById(R.id.event_type)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.drowsiness_event_item, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, @SuppressLint("RecyclerView") position: Int) {
        val event = events[position]
        val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
        val timeFormat = SimpleDateFormat("hh:mm a", Locale.getDefault())

        // Set text for textViews
        holder.dateText.text = dateFormat.format(event.timestamp)
        holder.timeText.text = timeFormat.format(event.timestamp)
        holder.durationText.text = event.getDurationText()
        holder.typeText.text = event.getType()

        when {
            position % 3 == 0 -> holder.cardView.setCardBackgroundColor(holder.itemView.context.getColor(R.color.drowsy_severe))
            position % 3 == 1 -> holder.cardView.setCardBackgroundColor(holder.itemView.context.getColor(R.color.drowsy_moderate))
            else -> holder.cardView.setCardBackgroundColor(holder.itemView.context.getColor(R.color.drowsy_mild))
        }

        if (position > lastPosition) {
            val animation = AnimationUtils.loadAnimation(holder.itemView.context, R.anim.item_animation_from_right)
            holder.itemView.startAnimation(animation)
            lastPosition = position
        }
    }

    override fun getItemCount() = events.size
}