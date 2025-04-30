import socket from "./socket";
import { FrameData, PredictionData } from "./types";

// 소켓 연결 초기화
export const initSocketConnection = (): void => {
  socket.connect();

  socket.on("connect", () => {
    console.log("✅ 소켓 연결 성공:", socket.id);
  });

  socket.on("disconnect", () => {
    console.log("❌ 소켓 연결 끊김");
  });
};

// 프레임 데이터 보내기
export const sendFrame = (frameData: FrameData): void => {
  if (socket.connected) {
    socket.emit("frame", frameData);
    console.log("📤 프레임 전송:", frameData);
  } else {
    console.warn("소켓이 연결되지 않았습니다. frame 전송 실패");
  }
};

// 서버로부터 예측 결과 받기
export const onPrediction = (callback: (data: PredictionData) => void): void => {
  socket.on("prediction", (data) => {
    console.log("📈 예측 결과 수신:", data);
    callback(data);
  });
};