package com.joeyv23.vortex;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjectionManager;
import android.util.Log;

import androidx.activity.result.ActivityResult;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;

@CapacitorPlugin(
    name = "AudioCapture",
    permissions = {
        @Permission(strings = { Manifest.permission.RECORD_AUDIO }, alias = "audio")
    }
)
public class AudioCapturePlugin extends Plugin {

    private static final String TAG = "AudioCapturePlugin";
    private static final int WS_PORT = 8765;

    @PluginMethod
    public void startCapture(PluginCall call) {
        Activity activity = getActivity();
        if (activity == null) {
            call.reject("Activity not available");
            return;
        }

        // Check RECORD_AUDIO permission (required for AudioPlaybackCapture)
        if (ContextCompat.checkSelfPermission(getContext(), Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissionForAlias("audio", call, "onAudioPermissionResult");
            return;
        }

        launchProjectionRequest(call);
    }

    @com.getcapacitor.annotation.PermissionCallback
    private void onAudioPermissionResult(PluginCall call) {
        if (getPermissionState("audio") == com.getcapacitor.PermissionState.GRANTED) {
            launchProjectionRequest(call);
        } else {
            call.reject("Microphone permission is required for system audio capture");
        }
    }

    private void launchProjectionRequest(PluginCall call) {
        Activity activity = getActivity();
        MediaProjectionManager mpm = (MediaProjectionManager)
                activity.getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        Intent intent = mpm.createScreenCaptureIntent();
        startActivityForResult(call, intent, "onProjectionResult");
    }

    @ActivityCallback
    private void onProjectionResult(PluginCall call, ActivityResult result) {
        if (call == null) {
            Log.e(TAG, "PluginCall was null in callback");
            return;
        }

        if (result.getResultCode() != Activity.RESULT_OK) {
            call.reject("User denied screen capture permission");
            return;
        }

        Intent data = result.getData();
        if (data == null) {
            call.reject("No projection data returned");
            return;
        }

        // Start the foreground service with the projection token
        Intent serviceIntent = new Intent(getContext(), AudioCaptureService.class);
        serviceIntent.setAction(AudioCaptureService.ACTION_START);
        serviceIntent.putExtra(AudioCaptureService.EXTRA_RESULT_CODE, result.getResultCode());
        serviceIntent.putExtra(AudioCaptureService.EXTRA_RESULT_DATA, data);
        serviceIntent.putExtra(AudioCaptureService.EXTRA_WS_PORT, WS_PORT);
        getContext().startForegroundService(serviceIntent);

        JSObject ret = new JSObject();
        ret.put("port", WS_PORT);
        ret.put("status", "capturing");
        call.resolve(ret);
    }

    @PluginMethod
    public void stopCapture(PluginCall call) {
        Intent serviceIntent = new Intent(getContext(), AudioCaptureService.class);
        serviceIntent.setAction(AudioCaptureService.ACTION_STOP);
        getContext().startService(serviceIntent);

        JSObject ret = new JSObject();
        ret.put("status", "stopped");
        call.resolve(ret);
    }

    @PluginMethod
    public void isCapturing(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("capturing", AudioCaptureService.isRunning());
        call.resolve(ret);
    }
}
