# 🥇 논문 : 2학기 ~ 겨울방학
### :star: 주제 : 자율 주행 로봇의 경로 회피계획 알고리즘
<h3>1. 데이터 셋</h3>

- 자율 주행 : KITTI Vision Benchmark Suite
- 저작권 : Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License

<h3>2. 방법론 (이론 정리)</h3>
<h4>🥈 깊이 영상은 사용하지 않을 예정 (일반 RGB 사진과 Lidar 데이터만)</h4><br>

**1단계 : 움직이는 대상 검출**
- 전방 사진 / Lidar 사진 분석 (대상 3D 박스로 검출과 Lidar 사진과의 결합)
- Lidar 데이터의 3차원 공간 맵핑 (대상 검출 포함)

**2단계 : 1차 경로 계획**
- 내가 가는 경로의 설정 (구체적인 좌표값혹은 예상으로 계산), 벡터
- 검출된 대상을 생각하며 예상 경로 생성

**3단계 : 2차 경로 맵핑**
- 움직이는 대상을 상대속도를 고려하여 같은 경로, 벡터 생성 (단 무한대로 확장하지 않음)
- 움직이는 대상의 경로와 자율주행차의 경로 계산, 1차 경로 설계의 보완 개념으로 접근 (즉, 1차 경로는 유지하지만 시간별로 조금씩만 조정하는 느낌)
- 기본 적으로 미세 회피 경로 계획만 고려하며, 회피 경로가 정지되는 경우는 일단 고려하지 않음 <br>
##
### :star: 맵핑 경로 설계 알고리즘 탐색중
<pre><code> 1. TEB Planner (Timed Elastic Band): 이 알고리즘은 경로의 곡률을 고려하여 차량이 실시간으로 유연한 경로를 따라갈 수 있도록 합니다. 
  특히, Lidar를 통해 주변 환경을 파악하여 장애물을 회피하면서 효율적인 경로를 찾습니다. 이는 자율주행 차량에서 많이 사용되는 로컬 플래닝 방법 중 하나입니다​.
  
2. Dynamic Window Approach (DWA): 이 알고리즘은 차량의 속도와 가속도를 고려하여 이동 가능한 경로를 실시간으로 계산하며, 
  주로 Lidar를 활용해 장애물을 감지하고 회피합니다​.
  
3. Hybrid A* 알고리즘: 이 방법은 A* 알고리즘을 확장한 것으로, 차량의 동적 제약 조건을 고려하여 최적의 경로를 탐색합니다. 
  특히, 장애물이 많은 복잡한 환경에서 효율적입니다. Reed-Shepp 확장을 통해 회전 반경을 고려한 경로 탐색도 가능합니다​.
  
4. Voronoi Field: 이 방법은 장애물과의 거리를 계산하여 안전한 경로를 생성합니다. 
  좁은 공간에서도 경로를 유지할 수 있는 장점이 있어, 주차나 복잡한 환경에서 유용하게 사용됩니다​.
</code></pre>

##
### 🏴 연구 진행 현황
### 1단계 : 3D 바운딩 박스의 대상 검출하기
Lidar 데이터를 2차원 깊이 영상으로 바꾼후 YOLO 검출, 이후 검출 데이터를 토대로 레이다에 맵핑하기 ✔️
- [깊이 영상 참고 사이트](https://velog.io/@happy_quokka/Sensor-Fusion-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-3.-LiDAR-Projection-%EA%B3%BC%EC%A0%95-%EB%B0%8F-%EA%B5%AC%ED%98%84-Open3D-Mayavi)
- [KITTI 연구 대상 사이트](https://github.com/kuixu/kitti_object_vis?tab=readme-ov-file)

데이터셋 연구중

- [complex-yolov4](https://www.youtube.com/watch?v=F3IEobi7Li4)
- [YOLO + DeepSORT : 경로 설계](https://www.youtube.com/watch?v=w80cToFgto8) <br><br>
##
### :star: 주제 : 자율 주행 로봇의 경로 회피계획 알고리즘 (2D 전방 이미지 기반 솔류션)
<h3>1. 데이터 셋</h3>

- 학습 데이터 셋 : Cityspace (픽셀 학습)
- 테스트 데이터 셋 : KITTI

<h3>2. 방법론 (이론 정리)</h3>

1. 이미지 전처리 (OpenCV를 통한 엣지추출)
   - 컨투어 선명하게 하기
2. 1차 경로 계획 (세그먼트 아키텍쳐 와 OpenCV)
   - U-net + LSTM 아키텍쳐 모델 이용 : Cityspace 픽셀 학습 후 세그먼트 분류 vs DeepLab
   - 세그먼트로 경로 와 대상 추출 후, 경로 클래스만 표시 (경로는 표시안된 대상을 기준으로 생성)
   - 경로에서의 자율 주행 로봇의 이동 경로는 OpenCV로 나타내기 (직선으로 표시) / 이때 세그먼트 경로 클래스는 "안전거리"
3. 대상 검출과 추척및 속도 벡터 계산
   - YOLO-DeeSORT 파이프라인으로 움직이는 대상을 검출하고 추적 : 1초당 이동거리를 통해 속도 벡터 예상 (OpenCV로 표시)
   - 예상된 속도 벡터는 그 대상의 이동 경로로 추정하며 벡터가 "안전거리"를 침범했을 떄 1차 경로 직선 경로를 수정
4. 2차 회피 경로 계획
   - 1차 경로 (직선)에서 안전거리 침범시 직성 경로를 실시간 수정

실험 예정
- U-net + LSTM의 IOC
- U-net과 LSTM과 DeepLab의 비교 (연산 시간과 경로 클래스 검출율, 안전거리 계산)
- YOLO-DeepSORT의 IOC, 성늘 지표
- 3,4의 시각화

<h3>연구 중</h3>
1. U-net 재학습 : https://www.youtube.com/watch?v=n4_ZuntLGjg
2. opencv : https://www.youtube.com/watch?v=iRTuCYx6quQ

