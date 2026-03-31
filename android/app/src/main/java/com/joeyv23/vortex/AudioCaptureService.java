package com.joeyv23.vortex;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioPlaybackCaptureConfiguration;
import android.media.AudioRecord;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.IBinder;
import android.util.Log;

import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class AudioCaptureService extends Service {

    private static final String TAG = "AudioCaptureService";
    private static final String CHANNEL_ID = "vortex_audio_capture";
    private static final int NOTIFICATION_ID = 1;
    private static final int SAMPLE_RATE = 44100;
    private static final int BUFFER_SIZE_SAMPLES = 1024;

    public static final String ACTION_START = "com.joeyv23.vortex.START_CAPTURE";
    public static final String ACTION_STOP = "com.joeyv23.vortex.STOP_CAPTURE";
    public static final String EXTRA_RESULT_CODE = "resultCode";
    public static final String EXTRA_RESULT_DATA = "resultData";
    public static final String EXTRA_WS_PORT = "wsPort";

    private static volatile boolean sRunning = false;

    private MediaProjection mediaProjection;
    private AudioRecord audioRecord;
    private Thread captureThread;
    private PcmWebSocketServer wsServer;
    private volatile boolean capturing = false;

    public static boolean isRunning() {
        return sRunning;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) return START_NOT_STICKY;

        String action = intent.getAction();
        if (ACTION_START.equals(action)) {
            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
            @SuppressWarnings("deprecation")
            Intent resultData = intent.getParcelableExtra(EXTRA_RESULT_DATA);
            int wsPort = intent.getIntExtra(EXTRA_WS_PORT, 8765);

            Notification notification = buildNotification();
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);

            startCapture(resultCode, resultData, wsPort);
        } else if (ACTION_STOP.equals(action)) {
            stopCapture();
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
        }

        return START_NOT_STICKY;
    }

    private void startCapture(int resultCode, Intent resultData, int wsPort) {
        MediaProjectionManager mpm = (MediaProjectionManager)
                getSystemService(MEDIA_PROJECTION_SERVICE);
        mediaProjection = mpm.getMediaProjection(resultCode, resultData);

        if (mediaProjection == null) {
            Log.e(TAG, "MediaProjection is null");
            stopSelf();
            return;
        }

        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                stopCapture();
                stopForeground(STOP_FOREGROUND_REMOVE);
                stopSelf();
            }
        }, null);

        // Configure AudioPlaybackCapture — capture MEDIA and GAME audio
        AudioPlaybackCaptureConfiguration captureConfig =
                new AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
                        .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
                        .addMatchingUsage(AudioAttributes.USAGE_GAME)
                        .build();

        int bufferSize = Math.max(
                AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT),
                BUFFER_SIZE_SAMPLES * 2  // int16 = 2 bytes
        );

        audioRecord = new AudioRecord.Builder()
                .setAudioPlaybackCaptureConfig(captureConfig)
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build())
                .setBufferSizeInBytes(bufferSize)
                .build();

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to initialize");
            stopSelf();
            return;
        }

        // Start WebSocket server
        wsServer = new PcmWebSocketServer(wsPort);
        wsServer.start();

        // Start reading PCM data
        capturing = true;
        sRunning = true;
        audioRecord.startRecording();

        captureThread = new Thread(() -> {
            short[] buffer = new short[BUFFER_SIZE_SAMPLES];
            while (capturing) {
                int read = audioRecord.read(buffer, 0, BUFFER_SIZE_SAMPLES);
                if (read > 0 && wsServer != null) {
                    // Convert int16 to float32 (normalized -1.0 to 1.0) for WebSocket
                    ByteBuffer bb = ByteBuffer.allocate(read * 4);
                    bb.order(ByteOrder.LITTLE_ENDIAN);
                    for (int i = 0; i < read; i++) {
                        bb.putFloat(buffer[i] / 32768.0f);
                    }
                    bb.flip();
                    wsServer.broadcast(bb);
                }
            }
        }, "AudioCaptureThread");
        captureThread.start();

        Log.i(TAG, "Audio capture started on ws://127.0.0.1:" + wsPort);
    }

    private void stopCapture() {
        capturing = false;
        sRunning = false;

        if (captureThread != null) {
            try {
                captureThread.join(2000);
            } catch (InterruptedException ignored) {}
            captureThread = null;
        }

        if (audioRecord != null) {
            try {
                audioRecord.stop();
                audioRecord.release();
            } catch (Exception ignored) {}
            audioRecord = null;
        }

        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }

        if (wsServer != null) {
            try {
                wsServer.stop(1000);
            } catch (InterruptedException ignored) {}
            wsServer = null;
        }

        Log.i(TAG, "Audio capture stopped");
    }

    private void createNotificationChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Audio Capture",
                NotificationManager.IMPORTANCE_LOW
        );
        channel.setDescription("Shows while Vortex is capturing system audio");
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(channel);
    }

    private Notification buildNotification() {
        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("Vortex")
                .setContentText("Visualizing system audio")
                .setSmallIcon(android.R.drawable.ic_media_play)
                .setOngoing(true)
                .build();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        stopCapture();
        super.onDestroy();
    }

    /**
     * Minimal WebSocket server that broadcasts PCM float32 frames to connected clients.
     */
    private static class PcmWebSocketServer extends WebSocketServer {

        PcmWebSocketServer(int port) {
            super(new InetSocketAddress("127.0.0.1", port));
            setReuseAddr(true);
        }

        @Override
        public void onOpen(WebSocket conn, ClientHandshake handshake) {
            Log.i(TAG, "WebSocket client connected: " + conn.getRemoteSocketAddress());
        }

        @Override
        public void onClose(WebSocket conn, int code, String reason, boolean remote) {
            Log.i(TAG, "WebSocket client disconnected");
        }

        @Override
        public void onMessage(WebSocket conn, String message) {
            // Not expecting text messages from client
        }

        @Override
        public void onError(WebSocket conn, Exception ex) {
            Log.e(TAG, "WebSocket error", ex);
        }

        @Override
        public void onStart() {
            Log.i(TAG, "WebSocket server started on port " + getPort());
        }

        public void broadcast(ByteBuffer data) {
            for (WebSocket conn : getConnections()) {
                if (conn.isOpen()) {
                    conn.send(data);
                }
            }
        }
    }
}
