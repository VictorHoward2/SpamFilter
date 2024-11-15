package org.pytorch.demo;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

// AppCompactActivity cung cấp khả năng tương thích ngược với các phiên bản Android cũ hơn.
public class BaseModuleActivity extends AppCompatActivity {
    private static final int UNSET = 0;

    // Handler: gửi, xử lý các message hoặc runnable trên 1 thread cụ thể
    // => giúp giao tiếp giữa các thread, đặc biệt là giữa main threat (UI thread) và background thread.
    // HandlerThread: tạo ra thread riêng để xử lý các tác vụ nền mà không ảnh hưởng main thread
    protected HandlerThread mBackgroundThread;
    protected Handler mBackgroundHandler;

    protected Handler mUIHandler;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mUIHandler = new Handler(getMainLooper());
    }

    @Override
    protected void onPostCreate(@Nullable Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        final Toolbar toolbar = findViewById(R.id.toolbar);
        if (toolbar != null) {
            setSupportActionBar(toolbar);
        }
        startBackgroundThread();
    }

    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("ModuleActivity");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    @Override
    protected void onDestroy() {
        stopBackgroundThread();
        super.onDestroy();
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            Log.e(Constants.TAG, "Error on stopping background thread", e);
        }
    }

}
