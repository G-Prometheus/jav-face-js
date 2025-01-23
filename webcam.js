const container = document.querySelector("#container");
let faceMatcher;

// Tải model và train dữ liệu
async function init() {
    await Promise.all([
        faceapi.loadSsdMobilenetv1Model("/models"),
        faceapi.loadFaceRecognitionModel("/models"),
        faceapi.loadFaceLandmarkModel("/models"),
    ]);

    Toastify({
        text: "Tải xong model nhận diện!",
    }).showToast();

    const labels = ["DoNhatLinh", "DoMinhQuan", "Fukada Eimi", "NguyenThanhMinh", "Triệu Lệ Dĩnh", "Rina Ishihara", "Takizawa Laura", "Yua Mikami"];
    const faceDescriptors = [];

    for (const label of labels) {
        const descriptors = [];
        for (let i = 1; i <= 4; i++) {
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpeg`);
            const detection = await faceapi
                .detectSingleFace(image)
                .withFaceLandmarks()
                .withFaceDescriptor();
            descriptors.push(detection.descriptor);
        }
        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
        Toastify({
            text: `Training xong data của ${label}!`,
        }).showToast();
    }

    faceMatcher = new faceapi.FaceMatcher(faceDescriptors, 0.6);
    Toastify({
        text: "Hoàn tất train dữ liệu!",
    }).showToast();
}

// Xử lý camera
async function startCamera() {
    const video = document.createElement("video");
    video.autoplay = true;
    container.innerHTML = "";
    container.append(video);

    // Lấy luồng video từ camera
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    // Đợi video sẵn sàng
    video.addEventListener("loadeddata", () => {
        processVideo(video);
    });
}

// Nhận diện khuôn mặt trên video
async function processVideo(video) {
    const canvas = faceapi.createCanvasFromMedia(video);
    container.append(canvas);

    const size = {
        width: video.videoWidth,
        height: video.videoHeight,
    };
    faceapi.matchDimensions(canvas, size);

    // Vòng lặp xử lý từng khung hình
    setInterval(async () => {
        const detections = await faceapi
            .detectAllFaces(video)
            .withFaceLandmarks()
            .withFaceDescriptors();
        const resizedDetections = faceapi.resizeResults(detections, size);

        // Xóa canvas cũ
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

        // Vẽ hộp nhận diện
        resizedDetections.forEach((detection) => {
            const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
            const box = new faceapi.draw.DrawBox(detection.detection.box, {
                label: bestMatch.toString(),
            });
            box.draw(canvas);
        });
    }, 100); // Xử lý mỗi 100ms
}

// Khởi tạo
init().then(() => {
    startCamera();
});
