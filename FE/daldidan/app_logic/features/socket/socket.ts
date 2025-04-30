import { io, Socket } from "socket.io-client";
import { PredictionData } from "./types";

const SERVER_URL = "https://browsing-thee-optimize-wonderful.trycloudflare.com";

// 타입 명시: socket은 서버와 PredictionData를 주고받는 Socket 객체
const socket: Socket = io(SERVER_URL, {
  transports: ["websocket"],
  autoConnect: false, // 수동으로 연결
});

export default socket;