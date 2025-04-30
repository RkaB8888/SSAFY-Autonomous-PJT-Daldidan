[프로젝트 기획회의]

아이디어 기획

### 주제 후보1
1. 구급대와 병원 간 이송을 지원하는 응급 의료 서비스
    1. 구급대원 용
        1. 구급대원에게 현재 위치를 기반으로 가장 환자의 증상과 적합하면서도 가까운 응급실 추천
        2. 환자의 증상과 응급실에 수용 가능한 과와 병상 수를 종합적으로 계산
    2. 병원 용
        1. 구급대원과 소통하며 응급실 환자 이송 여부 결정
        2. 응급 이송 통계 데이터 제공
    3. 데이터
        1. 실시간 응급실 가용병상정보 및 중증질환 수용가능정보, 위치정보 등 조회 가능
   4. AI모델 활용
       1. BioBERT, ClinicalBERT, KorBERT, KoELECTRA
           1. BioBERT

### 주제후보 2
1. 주식 모의 시뮬레이션(과거)
    1. 주기능
        1. 과거의 순간으로 돌아가서, 본인의 투자 전략이 적절한지 확인할 수 있음 → 프로그램 매매 시 해당 프로그램의 손익 계산?
    2. 기대효과
        1. 올바른 투자 전략 생성
        2. 투자 능력 향상
    3. 분석 방법
        1. 기본 알고리즘?
        2. AI모델 활용(학습)
            1. AI 기반 투자 리포트 생성
            2. 이미 학습된 모델을 가져와서, 추가 학습을 할 예정: 파인튜닝( 금융 뉴스/공시/리포트/주가 데이터)
                1. FinGPT
                2. FinBERT


[250416]
### 아이디어 구체화
1. 새로운 아이디어들 및 구체화
- 헬스 기반 식물 가드닝
  헬스 상태 기반 식물 감정 표현
  AI 기반 개인 맞춤 가드닝 피드백
  감정/헬스 일지 자동 기록 및 분석
- AI 콜봇을 활용해 은행 고객의 전화 문의를 자연어로 자동 응대하는 스마트 상담 시스템 개발
  자연어 기반 음성 인식 & 의도 파악
  대화형 응답 & FAQ 자동 처리
  상담 연결 최적화 (스마트 라우팅)

[250417]
### 주제 선정 및 아이디어 구체화
- 주제: 정책 정보 알림/ 청소년, 청년 등 범위 한정
  개인화된 정책 분석 
  정책 기조 지식 및 정보 제공 시스템
  민주주의 및 투표 제도 교육 플랫폼
  정책을 카테고리별(출산, 영유아, 청소년 교육 등)로 한정하여 정리하여 제공
- 향후 발전: 블록체인 기반 정책 약속 검증 시스템

[250421]
### 주제 선정 및 회의(*이 선정)
- 탐험용 지도
- 배달비 지도
- 소리 최적화 피드백 서비스
- 중고 책 거래 플랫폼 *
- Popzone *
- TrailTalk
- Groo *
- 오렌지 오렌지

[250422]
### 주제 선정 및 구체화 진행
- 실시간 과일 당도 측정
  
- 실시간 영상 처리

다중 색상 공간 변환을 한다. RGB를 활용하며, 특징을 추출할 수 있다.
기본적인 과일의 이미지를 학습하여, 그 과일에 대한 당도를 측정하는 것을 
기본 mvp로 한다.

- 기술적 장애물 및 해결 방안

조명 변화의 문제가 있으며, 품종간의 차이, 실시간 성능 한계가
기술적 장애물로 예상되며, 추후에 해결방안을 생각해보겠습니다. 
  

[250423]
- 주제 확정 및 코드 분석

- Mask-R-CNN 모델: 
하나의 이미지에서 객체를 찾아내고,
그게 어떤 클래스인지 분류하고,그 객체의 정확한 윤곽(픽셀 단위)을 마스크로 추출하는
다기능 딥러닝 모델

- 데이터셋 ⇒ 전북 장수 사과 당도 품질 데이터(50만 장의 사과 사진, 당도 측정 결과)
- 폰 실시간 영상 → Back 전달 과정(socket)


[250424]
- 모델 테스트 및 발표 PPT 작성
모델에 관련된 요구사항들을 설치하고, 사진들을 가지고 
테스트를 하여 모델의 적합성을 판단한다

- 발표 내용에 대한 자료조사를 바탕으로 PPT작성을 하였습니다.

[250425]
- 중간 발표 및 피드백
우리프로젝트의 mvp: 사과의 당도를 체크해서, 실시간으로 사용자에게 제공한다.

- 피드백: 사과와 수박까지 가능하도록 한다.
- 당도측정의 정확도는 70퍼 정도+ 사용자/이용자의 편의성을 많이 고려해서 프론트엔드를 설계하라.

- 현재 사용을 하고 있는 사용자들의 시간을 더욱 단축하고, 사용자의 편의를 제공하고자 노력한다

- 앞으로 다양한 데이터 셋, 알고리즘 등을 구현하기 위해 많은 레퍼런스들을 찾아보자

[250428]
- ai모델의 구체화
1. 전반적으로 모든 과일들의 색상, 질감, 특징 등을 출력할 수 있는 알고리즘 코드를 작성하자
2. 각 과일에 대한 근거 논문 및 자료 들을 찾아본다
```
import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from feature_extractor import extract_gloss, extract_color_ratio, extract_texture
from utils import preprocess_image

# 1. 사진 폴더 경로
image_folder = "sample_images/"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 2. Sweetness Score 계산
sweetness_scores = []

for file_name in image_files:
    file_path = os.path.join(image_folder, file_name)
    
    # 이미지 읽기
    image = cv2.imread(file_path)
    
    # 전처리
    preprocessed_image = preprocess_image(image)
    
    # 특징 추출
    gloss = extract_gloss(preprocessed_image)
    color = extract_color_ratio(preprocessed_image)
    texture = extract_texture(preprocessed_image)
    
    # Sweetness Score 계산
    sweetness_score = (gloss * 0.4) + (color * 0.4) - (texture * 0.2)
    
    sweetness_scores.append(sweetness_score)

print("Sweetness Scores:", sweetness_scores)

# 3. Sweetness Score → 등급(A/B) 매칭해서 Brix 만들기
grades = ['A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'B', 'A', 'B', 'A']  # 네가 가지고 있는 등급 순서대로
brix_values = []
for grade in grades:
    if grade == 'A':
        brix_values.append(14.5)
    else:
        brix_values.append(13.0)

# 4. numpy 변환
sweetness_scores = np.array(sweetness_scores).reshape(-1, 1)
brix_values = np.array(brix_values)

# 5. 선형 회귀 학습
model = LinearRegression()
model.fit(sweetness_scores, brix_values)

# 6. 결과 출력
print("회귀식: Brix = {:.3f} * Score + {:.3f}".format(model.coef_[0], model.intercept_))

```
3. 인프라 기본 설정을 학습 및 인프라를 직접 구현.

[250429]인프라 기본설정 및 ai모델 고도화방안 고려

### 인프라 기본설정 
1. termius 설정
2. termius에서 작업을 해보자
```bash
#도커 설치 명령어
Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```bash
일반 사용자 권한 설정
sudo usermod -aG docker $USER

등록 && 시작
sudo systemctl enable docker
sudo systemctl start docker 
sudo systemctl status docker
#status로 아래 사진의 결과값을 보여줄 수 있다.
```

- apt == advanced package tool를 뜻함.


4. 엔지닉스
```bash
sudo apt install curl gnupg2 ca-certificates lsb-release ubuntu-keyring
sudo apt install nginx
sudo systemctl enable nginx
sudo systemctl start nginx 
sudo systemctl status nginx
```

5. 방화벽 설정
6. HTTPS관련 설정인 ssl인증서 발급
```bash
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx

적용할 도메인 주소와 이메일 입력해주기
sudo certbot certonly --nginx -d {도메인주소}
-> 에서 나오는 내용을----if위의 내용을 복사해서  gpt에게 물어보자!!!! 
질문내용 "이거에 맞게 ssl적용되는 nginx.conf를 만들어줘"
```

7. 자바 설치
   백엔드를 FastAPI로 쓸예정이지만, 예전플젝을 참고하여 springboot가 추가될지도 모르는 상황에 대비하기위해,

자바 버전을 예전플젝버전을 사용할 예정으로 생각하고, 버전을 알아오니, 17이었다

8. 자바 설치했으니,,, 환경변수 설정을 해보자…
```bash
echo $JAVA_HOME

sudo nano /etc/environment
 #environment를 써야 젠킨스 가능. 그냥 bashsr?? 은 젠킨스가 무시할 수 있음.
 
 #안의 내용을 이렇게 추가한다.
PATH="~~~~:/usr/lib/jvm/java-17-openjdk-amd64/bin"
JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
```

9. 젠킨스 설치
- 젠킨스 설치를 해보자...!

### ai모델 고도화방안 고려
- 고도화 구상

#### 광택(Glossiness)
- 단순 밝기 평균이 아니라, 빛의 "하이라이트 영역(반사 강한 부분)" 비율 계산

#### 색상(Color)
- RGB가 아니라 HSV 색공간 기준으로 Hue(색상) 분포를 분석

#### 텍스처(Texture)
- Gray Level Co-occurrence Matrix (GLCM) 기반의 콘트라스트, 에너지, 동질성(Homogeneity) 계산


```
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ----------------------------------------------------------
# 광택 (Glossiness) 추출 함수
# → 이미지의 밝은 영역(하이라이트) 비율을 계산
# ----------------------------------------------------------
def extract_gloss(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 밝기가 220 이상인 픽셀 수를 계산
    bright_pixels = np.sum(gray > 220)
    
    # 전체 픽셀 수
    total_pixels = gray.size
    
    # 광택 비율 (밝은 픽셀 수 / 전체 픽셀 수)
    glossiness = bright_pixels / total_pixels
    return glossiness

# ----------------------------------------------------------
# 색상 (Color) 추출 함수 (HSV 색공간 기반)
# → 색조(Hue), 채도(Saturation), 명도(Value) 평균값 추출
# ----------------------------------------------------------
def extract_color_hsv(image):
    # 이미지를 HSV 색공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Hue(색조) 평균 (0~180 범위를 0~1로 정규화)
    mean_hue = np.mean(hsv[:, :, 0]) / 180
    
    # Saturation(채도) 평균 (0~255 범위를 0~1로 정규화)
    mean_saturation = np.mean(hsv[:, :, 1]) / 255
    
    # Value(명도) 평균 (0~255 범위를 0~1로 정규화)
    mean_value = np.mean(hsv[:, :, 2]) / 255
    
    return mean_hue, mean_saturation, mean_value

# ----------------------------------------------------------
# 텍스처 (Texture) 추출 함수 (GLCM 기반)
# → 텍스처의 대비(Contrast)와 동질성(Homogeneity) 계산
# ----------------------------------------------------------
def extract_texture(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GLCM (Gray-Level Co-occurrence Matrix) 계산
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    
    # 대비 (Contrast) 특징 추출
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    # 동질성 (Homogeneity) 특징 추출
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return contrast, homogeneity

```

[250430]
- 젠킨스 설치 및 https설정
인프라 를 완료하고, 백엔드 CI/CD를 생각해보았습니다.

- AI모델을 좀더 고도화하는 방법을 찾아보고 있습니다.

- 직접 모델을 만드는 방법 또는 CNN회귀모델을 활용하여 학습하는 방법을 고려하고 있습니다.
- 정확도와 입력값들을 고려하여 고도화할 예정입니다.