from flask import Flask, render_template, Response
import cv2
import numpy as np 

app = Flask(__name__)
cap = cv2.VideoCapture(0)
imgTarget = cv2.imread("./static/targetpic.png")
myVid = cv2.VideoCapture('./static/vid.mp4')

detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo=cv2.resize(imgVideo, (wT, hT))

orb= cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)

def gen_frames():  
    while True:
        success, imgWebCam = cap.read()  # read the camera frame
        if not success:
            break
        else:
            global detection, frameCounter, hT, wT, cT, imgVideo, imgTarget, myVid
            imgAug = imgWebCam.copy()
            kp2, des2= orb.detectAndCompute(imgWebCam, None)
            #imgWebCam = cv2.drawKeypoints(imgWebCam, kp2, None)

            if detection == False:
                myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frameCounter=0
            else:
                if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
                    myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frameCounter=0
                success, imgVideo = myVid.read()
                imgVideo=cv2.resize(imgVideo, (wT, hT))

                

            bf = cv2.BFMatcher()
            matches=bf.knnMatch(des1, des2, k=2)
            good=[]
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)


            imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebCam, kp2, good, None, flags=2 )

            if len(good) >30:
                detection = True
                srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

                matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
            
                pts= np.float32([[0,0], [0,hT], [wT, hT], [wT, 0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,matrix)
                img2 = cv2.polylines(imgWebCam, [np.int32(dst)], True, (255, 0, 255), 3)

                imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebCam.shape[1], imgWebCam.shape[0]))

                maskNew = np.zeros((imgWebCam.shape[0], imgWebCam.shape[1]), np.uint8)
                cv2.fillPoly(maskNew,[np.int32(dst)], (255,255,255) )
                maskInv = cv2.bitwise_not(maskNew)
                imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
                imgAug = cv2.bitwise_or(imgAug, imgWarp)


            ret, buffer = cv2.imencode('.jpg', imgAug)
            frameCounter+=1
            imgAug = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + imgAug + b'\r\n')  # concat frame one by one and show result



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')   
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)