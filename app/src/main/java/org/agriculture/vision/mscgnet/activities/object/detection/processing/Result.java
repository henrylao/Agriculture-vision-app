package org.agriculture.vision.mscgnet.activities.object.detection.processing;

import android.graphics.Rect;

public class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
}