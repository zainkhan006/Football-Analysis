import cv2
import numpy as np
import config
import dataLoader
from homography import PitchHomography
from refine_homography import build_line_mask, refine_homography, sample_canonical_pitch


def loadAnalyticsH(seqName):
    """loads the saved per-clip homography from homographies/<seqName>.npz"""
    homoPath = config.projectRoot / "homographies" / f"{seqName}.npz"
    if not homoPath.exists():
        raise FileNotFoundError(f"no analytics homography at {homoPath}")
    return PitchHomography.load(homoPath)


def refinePerFrame(frame, seedH, samples, gtBoxes, maxPx=30.0):
    """runs line-dt refinement on a single frame using seedH as starting point"""
    # mask out top 25% (ad boards) and player bboxes
    maskedFrame = frame.copy()
    cutoff = int(frame.shape[0] * 0.25)
    maskedFrame[:cutoff, :] = 0

    pad = 10
    for trackId, x, y, w, h in gtBoxes:
        x0 = max(0, int(x) - pad)
        y0 = max(0, int(y) - pad)
        x1 = min(frame.shape[1], int(x + w) + pad)
        y1 = min(frame.shape[0], int(y + h) + pad)
        maskedFrame[y0:y1, x0:x1] = 0

    mask = build_line_mask(maskedFrame)
    try:
        refinedMat, info = refine_homography(
            seedH, mask, samples,
            max_px=maxPx,
            max_iter=30,
            prior_weight=5.0,
        )
        return refinedMat, True
    except Exception as e:
        print(f"  refinement failed: {e}")
        return seedH, False


def computePerFrameSequence(seqPath, analyticsH, verbose=False):
    """generator that yields per-frame homography matrices

    each yielded item is a dict with frameId, frame, hMat (3x3 world to pixel)
    """
    samples = sample_canonical_pitch(step_m=2.0)
    prevH = analyticsH.H_world_to_pixel.copy()

    for data in dataLoader.loadSequence(seqPath):
        frameId = data["frameId"]
        frame = data["frame"]

        refinedH, ok = refinePerFrame(frame, prevH, samples, data["boxes"])

        if ok:
            prevH = refinedH

        else:
            if verbose:
                print(f"  frame {frameId} used previous frame H as fallback")

        yield {
            "frameId": frameId,
            "frame": frame,
            "hMat": prevH.copy(),
        }


def drawPitchOverlay(frame, hMat):
    """draws the canonical pitch model on the frame using hMat"""
    from pitch_config import DEFAULT_PITCH
    out = frame.copy()

    # pitch outline
    outline = np.array(DEFAULT_PITCH.outline_m, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(outline, hMat).reshape(-1, 2).astype(int)
    cv2.polylines(out, [projected], isClosed=True, color=(0, 255, 0), thickness=2)

    # halfway line
    L = DEFAULT_PITCH.length
    W = DEFAULT_PITCH.width
    halfway = np.array([[(L / 2, 0)], [(L / 2, W)]], dtype=np.float32)
    projectedHalf = cv2.perspectiveTransform(halfway, hMat).reshape(-1, 2).astype(int)
    cv2.line(out, tuple(projectedHalf[0]), tuple(projectedHalf[1]), (255, 200, 0), 2)

    # centre circle
    r = DEFAULT_PITCH.centre_circle_radius
    angles = np.linspace(0, 2 * np.pi, 64)
    circleWorld = np.array([
        [(L / 2 + r * np.cos(a), W / 2 + r * np.sin(a)) for a in angles]
    ], dtype=np.float32).reshape(-1, 1, 2)
    projectedCircle = cv2.perspectiveTransform(circleWorld, hMat).reshape(-1, 2).astype(int)
    cv2.polylines(out, [projectedCircle], isClosed=True, color=(0, 255, 255), thickness=2)

    return out


if __name__ == "__main__":
    seqName = config.testSequence
    print(f"loading analytics homography for {seqName}")
    analyticsH = loadAnalyticsH(seqName)
    print(f"  calibration frame {analyticsH.source_frame_id}, reproj err {analyticsH.fit_error_px:.2f}px")

    sampleFrameIds = [1, 100, 400, 700, 888]
    print(f"computing per-frame homography, saving samples at frames {sampleFrameIds}")

    saved = {}
    for data in computePerFrameSequence(config.testSequencePath, analyticsH, verbose=True):
        if data["frameId"] in sampleFrameIds:
            overlay = drawPitchOverlay(data["frame"], data["hMat"])
            cv2.putText(overlay, f"frame {data['frameId']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            outPath = config.outputsDir / f"per_frame_h_{data['frameId']:04d}.jpg"
            cv2.imwrite(str(outPath), overlay)
            saved[data["frameId"]] = outPath
            print(f"  saved {outPath.name}")

        if data["frameId"] >= max(sampleFrameIds):
            break

    print(f"saved {len(saved)} sample frames to outputs")
    print("open each to verify the projected pitch tracks the camera motion")