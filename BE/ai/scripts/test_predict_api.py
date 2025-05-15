import base64
import requests
import tkinter as tk
from tkinter import filedialog

# Tkinter 루트 윈도우 생성 (보이지 않게 숨김)
root = tk.Tk()
root.withdraw()

# 1. 파일 다이얼로그로 이미지 선택
file_path = filedialog.askopenfilename(
    title="당도 추론용 이미지를 선택하세요",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")],
)

if not file_path:
    print("이미지를 선택하지 않았습니다.")
    exit()

# 2. 이미지 → base64 인코딩
with open(file_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

# 3. 요청 구성
url = "https://k12e206.p.ssafy.io/predict"
data = {
    "id": 1,
    "image_base64": encoded,
}

# 4. 요청 전송
response = requests.post(url, data=data)

# 5. 응답 출력
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
