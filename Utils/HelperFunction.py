import cv2

# helper function to render video from agent playoffs
def renderVideo(outName=str, frameDict=dict, fps=int):
    output_file = outName + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = fps
    frame_test = frameDict[0]
    frame_size = (frame_test.shape[1], frame_test.shape[0])

    vidout = cv2.VideoWriter(output_file, fourcc=fourcc, fps=fps, frameSize=frame_size)
    for frame in range(len(frameDict)):
        bgr_frame = cv2.cvtColor(frameDict[frame], cv2.COLOR_RGB2BGR)
        vidout.write(bgr_frame)
    vidout.release()
    cv2.destroyAllWindows()
