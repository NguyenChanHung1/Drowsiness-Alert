package com.app.videobaseddrowsinessdetection

import android.os.Bundle
import android.view.View
import android.view.animation.AnimationUtils
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.app.videobaseddrowsinessdetection.adapters.EventAdapter
import com.app.videobaseddrowsinessdetection.utils.EventManager
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.snackbar.Snackbar

class HistoryActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var eventManager: EventManager
    private lateinit var adapter: EventAdapter
    private lateinit var emptyView: View
    private lateinit var clearAllFab: FloatingActionButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_drowsiness_history)

        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Drowsiness History"

        // Apply window insets
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.history_container)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Initialize views
        recyclerView = findViewById(R.id.drowsiness_events_recyclerview)
        emptyView = findViewById(R.id.empty_view)
        clearAllFab = findViewById(R.id.clear_all_fab)

        // Set up recycler view
        recyclerView.layoutManager = LinearLayoutManager(this)

        eventManager = EventManager(applicationContext)
        loadEvents()

        clearAllFab.setOnClickListener {
            showClearConfirmationDialog()
        }

        val animation = AnimationUtils.loadAnimation(this, R.anim.item_animation_from_bottom)
        recyclerView.startAnimation(animation)
    }

    private fun loadEvents() {
        val events = eventManager.getAllEvents()
        adapter = EventAdapter(events)
        recyclerView.adapter = adapter

        // Show empty view if there are no events
        if (events.isEmpty()) {
            recyclerView.visibility = View.GONE
            emptyView.visibility = View.VISIBLE
            clearAllFab.hide()
        } else {
            recyclerView.visibility = View.VISIBLE
            emptyView.visibility = View.GONE
            clearAllFab.show()
        }
    }

    private fun showClearConfirmationDialog() {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Clear Drowsiness History")
            .setMessage("Are you sure you want to clear all drowsiness history? This action cannot be undone.")
            .setPositiveButton("Clear") { _, _ ->
                eventManager.clearAllEvents()
                loadEvents() // Reload the empty list

                Snackbar.make(
                    findViewById(R.id.history_container),
                    "Drowsiness history cleared",
                    Snackbar.LENGTH_SHORT
                ).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }
}