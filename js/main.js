(async () => {
  const cardSize = { w: 910, h: 550 };
  const resolution = { w: 1080, h: 720 };
  const canvasSize = { w: 360, h: 240 };
  const video = document.getElementById("videoInput");
  let isPushed = false;

  media = navigator.mediaDevices
    .getUserMedia({
      audio: false,
      video: {
        height: canvasSize.h,
        width: canvasSize.w,
      },
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((err) => {
      console.log(err);
    });

  const canvasFrame = document.getElementById("canvasOutput"); // canvasFrame is the id of <canvas>
  const context = canvasFrame.getContext("2d");

  const button = document.getElementById("button");
  button.onclick = () => {
    isPushed = !isPushed;
  };

  let src = new cv.Mat(canvasSize.h, canvasSize.w, cv.CV_8UC4);
  let dst = new cv.Mat(canvasSize.h, canvasSize.w, cv.CV_8UC1);
  const FPS = 30;
  let last = new cv.Mat();
  let last2 = new cv.Mat();

  // ocr
  const doOCR = async () => {
    Tesseract.recognize(
      canvasRes,
      {
        lang: "eng",
        tessedit_pageseg_mode: "SINGLE_BLOCK",
      },
      { logger: (m) => console.log(m) }
    ).then((res) => {
      let text = res.text;
      // replace frequently missed errors
      text = text.replace(/ /g, "");
      document.getElementById("textRes").value = text;
    });
  };

  // capture video
  const processVideo = async () => {
    if (!last.empty()) {
      last.copyTo(last2);
    }
    let begin = Date.now();
    context.drawImage(video, 0, 0, canvasSize.w, canvasSize.h);
    src.data.set(context.getImageData(0, 0, canvasSize.w, canvasSize.h).data);

    // edge detection
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
    cv.threshold(dst, dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    let contours = new cv.MatVector();
    let poly = new cv.MatVector();
    let hierarchy = new cv.Mat();

    let res = [];

    cv.findContours(
      dst,
      contours,
      hierarchy,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_TC89_L1
    );

    let maxLevel = 0;
    for (let i = 0; i < contours.size(); i++) {
      let area = cv.contourArea(contours.get(i));
      if (area > 15000) {
        let tmp = new cv.Mat();
        cv.approxPolyDP(
          contours.get(i),
          tmp,
          0.01 * cv.arcLength(contours.get(i), true),
          true
        );
        if (tmp.rows === 4) {
          poly.push_back(tmp);
          for (let j = 0; j < 4; j++) {
            const [x, y] = tmp.intPtr(j);
            res.push([x, y]);
          }

          // decide the order of points
          res.sort((a, b) => a[0] * a[1] - b[0] * b[1]);
          const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            ...res[0],
            ...(res[1][1] > res[2][1] ? res[1] : res[2]),
            ...(res[1][1] > res[2][1] ? res[2] : res[1]),
            ...res[3],
          ]);
          const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0,
            0,
            0,
            cardSize.h,
            cardSize.w,
            0,
            cardSize.w,
            cardSize.h,
          ]);

          // fit it0
          const M = cv.getPerspectiveTransform(srcTri, dstTri);
          cv.warpPerspective(
            src,
            last,
            M,
            new cv.Size(cardSize.w, cardSize.h),
            cv.INTER_LINEAR,
            cv.BORDER_CONSTANT,
            new cv.Scalar()
          );

          // if (!last.empty() && !last2.empty()) {
          //   let img = last.convertTo(cv.cv_32F);
          //   // hist = cv.calcHist(last);
          //   // hist2 = cv.calcHist(last2);
          //   // console.log(hist.compareHist(hist2, cv.HISTCMP_CORREL));
          // }

          break;
        }
      }
    }

    for (let i = 0; i < poly.size(); i++) {
      cv.drawContours(
        src,
        poly,
        i,
        new cv.Scalar(255, 0, 0, 255),
        1,
        8,
        hierarchy,
        0
      );
    }

    cv.imshow("canvasOutput", src); // canvasOutput is the id of another <canvas>;
    // schedule next one.
    let delay = 1000 / FPS - (Date.now() - begin);
    const timeoutId = setTimeout(processVideo, delay);
    if (isPushed) {
      clearTimeout(timeoutId);
      let rect = new cv.Rect(20, 180, 800, 190);
      let mat = new cv.Mat();
      mat = last.roi(rect);
      cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY);
      cv.threshold(mat, mat, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
      cv.imshow("canvasRes", mat);
      document.getElementById("divOutput").hidden = true;
      document.getElementById("divRes").hidden = false;
      document.getElementById("divText").hidden = false;
      doOCR();
    }
  };
  // schedule first one.
  setTimeout(processVideo, 0);
  // cv.imshow("canvasOutput", last);
})();
